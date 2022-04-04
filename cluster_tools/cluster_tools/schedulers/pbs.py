"""Abstracts access to a PBS cluster via its command-line tools.
"""
import logging
import os
import re
from concurrent import futures
from typing import Dict, List, Optional, Tuple

from typing_extensions import Literal

from cluster_tools.util import call, chcall, random_string

from .cluster_executor import ClusterExecutor

# qstat vs. checkjob
PBS_STATES: Dict[str, List[str]] = {
    "Failure": [],
    "Success": [
        "C",  # Completed
        "F",  # It can have failed too, but we will notice this when we don't find the pickle file
    ],
    "Ignore": [
        "E",  # Job is exiting after having run
        "H",  # Job is held.
        "Q",  # job is queued, eligable to run or routed.
        "R",  # job is running.
        "T",  # job is being moved to new location.
        "W",  # job is waiting for its execution time
        "S",  # (Unicos only) job is suspend.
    ],
    "Unclear": [],
}


class PBSExecutor(ClusterExecutor):
    @classmethod
    def executor_key(cls) -> str:
        return "pbs"

    @staticmethod
    def get_job_array_index():
        return os.environ.get("PBS_ARRAYID", None)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("PBS_JOBID")

    @staticmethod
    def format_log_file_name(job_id_with_index, suffix=".stdout"):
        return "pbs.{}.log{}".format(str(job_id_with_index), suffix)

    @classmethod
    def get_job_id_string(cls):
        return cls.get_current_job_id()

    def submit_text(self, job):
        """Submits a PBS job represented as a job file string. Returns
        the job ID.
        """

        filename = self.get_temp_file_path(
            self.cfut_dir, "_temp_pbs_{}.sh".format(random_string())
        )
        with open(filename, "w") as f:
            f.write(job)
        jobid_desc, _ = chcall("qsub -V {}".format(filename))
        match = re.search("^[0-9]+", jobid_desc.decode("utf-8"))
        assert match is not None
        jobid = match.group(0)

        print("jobid", jobid)
        # os.unlink(filename)
        return int(jobid)

    def inner_submit(
        self,
        cmdline: str,
        job_name: Optional[str] = None,
        additional_setup_lines: Optional[List[str]] = None,
        job_count: Optional[int] = None,
    ) -> Tuple[List["futures.Future[str]"], List[Tuple[int, int]]]:
        """Starts a PBS job that runs the specified shell command line."""
        if additional_setup_lines is None:
            additional_setup_lines = []

        # if job_count is None else "$PBS_JOBID.$PBS_ARRAY_INDEX"
        # $PBS_JOBID will also include an array index if it's a job array
        log_path = self.format_log_file_path(self.cfut_dir, "$PBS_JOBID")
        print("log_path", log_path)

        job_resources_line = ""
        if self.job_resources is not None:
            specs = []
            for resource, value in self.job_resources.items():
                if resource == "time":
                    resource = "walltime"
                specs.append("{}={}".format(resource, value))
            if len(specs) > 0:
                job_resources_line = "#PBS -l {}".format(",".join(specs))

        job_array_line = ""
        if job_count is not None:
            if job_count == 1:
                # Even though, the t range is inclusive on both ends, pbs doesn't like 0-0 as a parameter.
                # Explicitly, listing the index works, though.
                job_array_line = "#PBS -t 0"
            else:
                job_array_line = "#PBS -t 0-{}".format(job_count - 1)

        script_lines = [
            "#!/bin/sh",
            "#PBS -j oe",  # join output and error stream
            # Apparently, it's important to have the -e line before -o
            "#PBS -e {}".format(log_path),
            "#PBS -o {}".format(log_path),
            '#PBS -N "{}"'.format(job_name),
            job_array_line,
            job_resources_line,
            *additional_setup_lines,
            "export PATH=$PBS_O_PATH",
            "cd $PBS_O_WORKDIR",
            "{}".format(cmdline),
        ]

        job_id = self.submit_text("\n".join(script_lines))
        job_id_future: "futures.Future[str]" = futures.Future()
        job_id_future.set_result(job_id)

        return [job_id_future], [(0, job_count or 1)]

    def check_for_crashed_job(
        self, job_id_with_index
    ) -> Literal["failed", "ignore", "completed"]:
        if len(str(job_id_with_index).split("_")) >= 2:
            a, b = job_id_with_index.split("_")
            job_id_with_index = f"{a}[{b}]"

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call("qstat -f {}".format(job_id_with_index))

        if exit_code != 0:
            logging.error(
                "Couldn't call checkjob to determine job's status. {}. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the pbs db. If this is the case, the process will hang forever.".format(
                    job_id_with_index
                )
            )
            return "ignore"
        else:

            job_state_search = re.search("job_state = ([a-zA-Z_]*)", str(stdout))
            if job_state_search:
                job_state = job_state_search.group(1)

                if job_state in PBS_STATES["Failure"]:
                    return "failed"
                elif job_state in PBS_STATES["Ignore"]:
                    return "ignore"
                elif job_state in PBS_STATES["Unclear"]:
                    logging.warning(
                        "The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(
                            job_id_with_index, job_state
                        )
                    )
                    return "ignore"
                elif job_state in PBS_STATES["Success"]:
                    return "completed"
                else:
                    logging.error("Unhandled pbs job state? {}".format(job_state))
                    return "ignore"
            else:
                logging.error(
                    "Could not extract pbs job state? {}...".format(stdout[0:10])
                )
                return "ignore"

    def get_pending_tasks(self):
        # Not implemented, yet. Currently, this is only used for performance optimization.
        return []
