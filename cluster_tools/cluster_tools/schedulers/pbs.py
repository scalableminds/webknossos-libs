"""Abstracts access to a PBS cluster via its command-line tools."""

import logging
import os
import re
from concurrent.futures import Future
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from cluster_tools._utils.call import call, chcall
from cluster_tools._utils.string_ import random_string
from cluster_tools.schedulers.cluster_executor import ClusterExecutor

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
        "Q",  # job is queued, eligible to run or routed.
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
    def get_job_array_index() -> Optional[int]:
        try:
            return int(os.environ["PBS_ARRAYID"])
        except KeyError:
            return None

    @staticmethod
    def get_current_job_id() -> str:
        r = os.environ.get("PBS_JOBID")
        assert r is not None
        return r

    @staticmethod
    def format_log_file_name(job_id_with_index: str, suffix: str = ".stdout") -> str:
        return f"pbs.{job_id_with_index}.log{suffix}"

    @classmethod
    def get_job_id_string(cls) -> str:
        return cls.get_current_job_id()

    def inner_handle_kill(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002 Unused method argument: `args`, kwargs
        scheduled_job_ids: List[Union[int, str]] = list(self.jobs.keys())

        if len(scheduled_job_ids):
            # Array jobs (whose id looks like `<job_id>_<array_index>`) don't need to be canceled individually,
            # but can be canceled together using the job_id.
            split_job_ids = map(lambda x: str(x).split("_"), scheduled_job_ids)
            # However array job ids need to include [] in the end.
            unique_job_ids = set(
                job_id_parts[0] if len(job_id_parts) == 1 else f"{job_id_parts[0]}[]"
                for job_id_parts in split_job_ids
            )
            # Send SIGINT signal instead of SIGTERM using qdel. This way, the jobs can
            # react to the signal, safely shutdown and signal (cancel) jobs they possibly scheduled, recursively.
            _stdout, stderr, exit_code = call(
                f"qsig -s SIGINT {' '.join(unique_job_ids)}"
            )

            if exit_code == 0:
                logging.debug(f"Canceled PBS jobs {', '.join(unique_job_ids)}.")
            else:
                logging.warning(
                    f"Couldn't automatically cancel all PBS jobs. Reason: {stderr}"
                )

    def submit_text(self, job: str) -> str:
        """Submits a PBS job represented as a job file string. Returns
        the job ID.
        """

        filename = self.get_temp_file_path(
            self.cfut_dir, "_temp_pbs_{}.sh".format(random_string())
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(job)
        jobid_desc, _ = chcall("qsub -V {}".format(filename))
        match = re.search("^[0-9]+", jobid_desc)
        assert match is not None
        jobid = match.group(0)

        print("jobid", jobid)
        # os.unlink(filename)
        return str(int(jobid))  # int() ensures coherent parsing

    def inner_submit(
        self,
        cmdline: str,
        job_name: Optional[str] = None,
        additional_setup_lines: Optional[List[str]] = None,
        job_count: Optional[int] = None,
    ) -> Tuple[List[Future[str]], List[Tuple[int, int]]]:
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
        job_id_future: Future[str] = Future()
        job_id_future.set_result(job_id)

        return [job_id_future], [(0, job_count or 1)]

    def check_job_state(
        self, job_id_with_index: str
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
            job_state_search = re.search("job_state = ([a-zA-Z_]*)", stdout)
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

    def get_pending_tasks(self) -> List:
        # Not implemented, yet. Currently, this is only used for performance optimization.
        return []
