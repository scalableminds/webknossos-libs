"""Abstracts access to a PBS cluster via its command-line tools.
"""
import re
import os
import threading
import time
from cluster_tools.util import chcall, random_string, local_filename, call
from .cluster_executor import ClusterExecutor
import logging
from typing import Union


# qstat vs. checkjob
PBS_STATES = {
    "Failure": [
    ],
    "Success": [
        "C",  # Completed
        "F"  # It can have failed too, but we will notice this when we don't find the pickle file
    ],
    "Ignore": [
        "E", # Job is exiting after having run
        "H", # Job is held.
        "Q", # job is queued, eligable to run or routed.
        "R", # job is running.
        "T", # job is being moved to new location.
        "W", # job is waiting for its execution time
        "S", # (Unicos only) job is suspend.
    ],
    "Unclear": [
    ]
}

def submit_text(job):
    """Submits a PBS job represented as a job file string. Returns
    the job ID.
    """

    filename = local_filename("_temp_{}.sh".format(random_string()))
    with open(filename, "w") as f:
        f.write(job)
    jobid_desc, _ = chcall("qsub -V {}".format(filename))
    match = re.search("^[0-9]+", jobid_desc.decode("utf-8") )
    assert match is not None    
    jobid = match.group(0)

    print("jobid", jobid)
    # os.unlink(filename)
    return int(jobid)


class PBSExecutor(ClusterExecutor):

    @staticmethod
    def get_job_array_index():
        return os.environ.get("PBS_ARRAYID", None)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("PBS_JOBID")

    def format_log_file_name(self, jobid):
        return local_filename("pbs.stdout.{}.log").format(str(jobid))

    def inner_submit(
        self,
        cmdline,
        job_name=None,
        additional_setup_lines=[],
        job_count=None,
    ):
        """Starts a PBS job that runs the specified shell command line.
        """

        # if job_count is None else "$PBS_JOBID.$PBS_ARRAY_INDEX"
        # $PBS_JOBID will also include an array index if it's a job array
        log_path = self.format_log_file_name("$PBS_JOBID")
        print("log_path", log_path)

        job_resources_line = ""
        if self.job_resources is not None:
            specs = []
            for resource, value in self.job_resources.items():
                if resource == "time":
                    resource == "walltime"
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
            "#PBS -j oe", # join output and error stream
            # Apparently, it's important to have the -e line before -o
            "#PBS -e {}".format(log_path),
            "#PBS -o {}".format(log_path),
            '#PBS -N "{}"'.format(job_name),
            job_array_line,
            job_resources_line,
            *additional_setup_lines,
            'export PATH=$PBS_O_PATH',
            'cd $PBS_O_WORKDIR',
            "{}".format(cmdline)
        ]

        return submit_text("\n".join(script_lines))


    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:
        if len(str(job_id).split("_")) >= 2:
            a, b = job_id.split("_")
            job_id = f"{a}[{b}]"

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call("qstat -f {}".format(job_id))

        if exit_code != 0:
            logging.error(
                "Couldn't call checkjob to determine job's status. {}. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the pbs db. If this is the case, the process will hang forever.".format(job_id)
            )
            return "ignore"
        else:

            job_state_search = re.search('job_state = ([a-zA-Z_]*)', str(stdout))
            if job_state_search:
                job_state = job_state_search.group(1)

                if job_state in PBS_STATES["Failure"]:
                    return "failed"
                elif job_state in PBS_STATES["Ignore"]:
                    return "ignore"
                elif job_state in PBS_STATES["Unclear"]:
                    logging.warn("The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(job_id, job_state))
                    return "ignore"
                elif job_state in PBS_STATES["Success"]:
                    return "completed"
                else:
                    logging.error("Unhandled pbs job state? {}".format(job_state))
                    return "ignore"
            else:
                logging.error("Could not extract pbs job state? {}...".format(stdout[0:10]))
                return "ignore"