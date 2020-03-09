"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import re
import os
import threading
import time
from cluster_tools.util import chcall, random_string, call
from .cluster_executor import ClusterExecutor
import logging
from typing import Union

SLURM_STATES = {
    "Failure": [
        "CANCELLED",
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "STOPPED",
        "TIMEOUT"
    ],
    "Success": [
        "COMPLETED"
    ],
    "Ignore": [
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "RESV_DEL_HOLD",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING"
    ],
    "Unclear": [
        "SUSPENDED",
        "REVOKED",
        "SIGNALING",
        "SPECIAL_EXIT",
        "STAGE_OUT"
    ]
}




class SlurmExecutor(ClusterExecutor):

    @staticmethod
    def get_job_array_index():
        return os.environ.get("SLURM_ARRAY_TASK_ID", None)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("SLURM_JOB_ID")

    def format_log_file_name(self, jobid):
        return "slurmpy.stdout.{}.log".format(str(jobid))

    def submit_text(self, job):
        """Submits a Slurm job represented as a job file string. Returns
        the job ID.
        """

        filename = self.get_temp_file_path("_temp_slurm{}.sh".format(random_string()))
        with open(filename, "w") as f:
            f.write(job)
        job_id, stderr = chcall("sbatch --parsable {}".format(filename))
        os.unlink(filename)

        if len(stderr) > 0:
            logging.warning(f"Submitting batch job emitted warnings: {stderr}")

        return int(job_id)


    def inner_submit(
        self,
        cmdline,
        job_name=None,
        additional_setup_lines=[],
        job_count=None,
    ):
        """Starts a Slurm job that runs the specified shell command line.
        """

        log_path = self.format_log_file_path("%j" if job_count is None else "%A_%a")

        job_resources_lines = []
        if self.job_resources is not None:
            for resource, value in self.job_resources.items():
                job_resources_lines += ["#SBATCH --{}={}".format(resource, value)]

        job_array_line = ""
        if job_count is not None:
            job_array_line = "#SBATCH --array=0-{}".format(job_count - 1)

        script_lines = (
            [
                "#!/bin/sh",
                "#SBATCH --output={}".format(log_path),
                '#SBATCH --job-name "{}"'.format(job_name),
                job_array_line
            ] + job_resources_lines
            + [*additional_setup_lines, "srun {}".format(cmdline)]
        )

        return self.submit_text("\n".join(script_lines))

    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:

        job_states = []

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call("scontrol show job {}".format(job_id))
        stdout = stdout.decode("utf8")

        if exit_code == 0:
            job_state_search = re.search('JobState=([a-zA-Z_]*)', str(stdout))
            if job_state_search:
                job_states = [job_state_search.group(1)]
            else:
                logging.error("Could not extract slurm job state? {}".format(stdout[0:10]))
        else:
            stdout, _, exit_code = call("sacct -j {} -o State -P".format(job_id))
            stdout = stdout.decode("utf8")

            if exit_code == 0:
                job_states = stdout.split("\n")[1:]

        if len(job_states) == 0:
            logging.error(
                "Couldn't call scontrol nor sacct to determine job's status. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
            )
            return "ignore"

        def matches_states(slurm_states):
            return len(list(set(job_states) & set(slurm_states))) > 0

        if matches_states(SLURM_STATES["Failure"]):
            return "failed"
        elif matches_states(SLURM_STATES["Ignore"]):
            return "ignore"
        elif matches_states(SLURM_STATES["Unclear"]):
            logging.warn("The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(job_id, job_states))
            return "ignore"
        elif matches_states(SLURM_STATES["Success"]):
            return "completed"
        else:
            logging.error("Unhandled slurm job state for job id {}? {}".format(job_id, job_states))
            return "ignore"

    def get_pending_tasks(self):
        try:
            # Get the job ids (%i) of the active user (-u) which are pending (-t) and format
            # them one-per-line (-r) while excluding the header (-h).
            stdout, _ = chcall("squeue -u $(whoami) -t PENDING -r -h --format=%i")
            stdout = stdout.decode("utf8")

            job_ids = set(stdout.split("\n"))
            return job_ids
        except Exception as e:
            logging.error("Couldn't query pending jobs. Polling for finished jobs might be slow.")
            return []
