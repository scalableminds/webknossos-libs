"""Abstracts access to a Slurm cluster via its command-line tools."""

import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from functools import lru_cache
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from cluster_tools._utils.call import call, chcall
from cluster_tools._utils.string_ import random_string
from cluster_tools.schedulers.cluster_executor import (
    NOT_YET_SUBMITTED_STATE,
    ClusterExecutor,
    RemoteException,
    RemoteOutOfMemoryException,
    RemoteResourceLimitException,
    RemoteTimeLimitException,
)

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
        "TIMEOUT",
    ],
    "Success": ["COMPLETED"],
    "Ignore": [
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "RESV_DEL_HOLD",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING",
    ],
    "Unclear": ["SUSPENDED", "REVOKED", "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT"],
}

SLURM_QUEUE_CHECK_INTERVAL = 1 if "pytest" in sys.modules else 60

T = TypeVar("T")


def noopDecorator(func: T) -> T:
    return func


cache_in_production = cast(
    Callable[[T], T], noopDecorator if "pytest" in sys.modules else lru_cache(maxsize=1)
)


class SlurmExecutor(ClusterExecutor):
    def __init__(
        self,
        debug: bool = False,
        keep_logs: bool = True,
        cfut_dir: str | None = None,
        job_resources: dict[str, Any] | None = None,
        job_name: str | None = None,
        additional_setup_lines: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            debug=debug,
            keep_logs=keep_logs,
            cfut_dir=cfut_dir,
            job_resources=job_resources,
            job_name=job_name,
            additional_setup_lines=additional_setup_lines,
            **kwargs,
        )
        self.submit_threads: list[_JobSubmitThread] = []

    @classmethod
    def executor_key(cls) -> str:
        return "slurm"

    @staticmethod
    def get_job_array_index() -> int | None:
        try:
            return int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            return None

    @staticmethod
    def get_job_array_id() -> str | None:
        try:
            return os.environ.get("SLURM_ARRAY_JOB_ID")
        except KeyError:
            return None

    @staticmethod
    def get_current_job_id() -> str:
        r = os.environ.get("SLURM_JOB_ID")
        assert r is not None
        return r

    @staticmethod
    def format_log_file_name(job_id_with_index: str, suffix: str = ".stdout") -> str:
        return f"slurmpy.{job_id_with_index}.log{suffix}"

    @classmethod
    def get_job_id_string(cls) -> str:
        job_id = cls.get_current_job_id()
        job_array_id = cls.get_job_array_id()
        job_array_index = cls.get_job_array_index()

        # This variable needs to be kept in sync with the job_id_string variable in the
        # inner_submit function.
        job_id_string = (
            job_id
            if job_array_id is None or job_array_index is None
            else f"{job_array_id}_{job_array_index}"
        )
        return job_id_string

    @staticmethod
    @cache_in_production
    def get_max_array_size() -> int:
        max_array_size_env = os.environ.get("SLURM_MAX_ARRAY_SIZE", None)
        if max_array_size_env is not None:
            logging.debug(
                f"SLURM_MAX_ARRAY_SIZE env variable specified which is {max_array_size_env}."
            )
            return int(max_array_size_env)

        max_array_size = 2**32
        # See https://unix.stackexchange.com/a/364615
        stdout, stderr, exit_code = call(
            "scontrol show config | sed -n '/^MaxArraySize/s/.*= *//p'"
        )
        if exit_code == 0:
            max_array_size = int(stdout)
            logging.debug(f"Slurm MaxArraySize is {max_array_size}.")
        else:
            logging.warning(
                f"Slurm's MaxArraySize couldn't be determined. Reason: {stderr}"
            )
        return max_array_size

    @staticmethod
    @cache_in_production
    def get_max_running_size() -> int:
        max_running_size_env = os.environ.get("SLURM_MAX_RUNNING_SIZE", None)
        if max_running_size_env is not None:
            logging.debug(
                f"SLURM_MAX_RUNNING_SIZE env variable specified which is {max_running_size_env}."
            )
            return int(max_running_size_env)

        return 0

    @staticmethod
    @cache_in_production
    def get_max_submit_jobs() -> int:
        max_submit_jobs_env = os.environ.get("SLURM_MAX_SUBMIT_JOBS", None)
        if max_submit_jobs_env is not None:
            logging.debug(
                f"SLURM_MAX_SUBMIT_JOBS env variable specified which is {max_submit_jobs_env}."
            )
            return int(max_submit_jobs_env)

        max_submit_jobs = 2**32
        # Check whether there is a limit per user
        stdout_user, stderr_user, _ = call(
            "sacctmgr list -n user $USER withassoc format=maxsubmitjobsperuser"
        )
        try:
            max_submit_jobs = int(stdout_user)
        except ValueError:
            # If there is no limit per user check whether there is a general limit
            stdout_qos, stderr_qos, _ = call(
                "sacctmgr list -n qos normal format=maxsubmitjobsperuser"
            )
            try:
                max_submit_jobs = int(stdout_qos)
            except ValueError:
                logging.warning(
                    f"Slurm's MaxSubmitJobsPerUser couldn't be determined. Reason: {stderr_user}\n{stderr_qos}"
                )
                return max_submit_jobs
        logging.debug(f"Slurm MaxSubmitJobsPerUser is {max_submit_jobs}.")
        return max_submit_jobs

    @staticmethod
    def get_number_of_submitted_jobs(state: str | None = None) -> int:
        number_of_submitted_jobs = 0
        state_string = f"-t {state}" if state else ""
        # --array so that each job array element is displayed on a separate line and -h to hide the header
        stdout, stderr, exit_code = call(
            f"squeue --array -u $USER -h {state_string} | wc -l"
        )

        job_state_string = f"with state {state} " if state else ""
        if exit_code == 0:
            number_of_submitted_jobs = int(stdout)
            logging.debug(
                f"Number of currently submitted jobs {job_state_string}is {number_of_submitted_jobs}."
            )
        else:
            logging.warning(
                f"Number of currently submitted jobs {job_state_string}couldn't be determined. Reason: {stderr}"
            )
        return number_of_submitted_jobs

    @classmethod
    def submit_text(cls, job: str, cfut_dir: str) -> str:
        """Submits a Slurm job represented as a job file string. Returns
        the job ID.
        """

        filename = cls.get_temp_file_path(cfut_dir, f"_temp_slurm{random_string()}.sh")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(job)

        # Workaround to avoid "Unable to satisfy cpu bind request" errors in nested slurm scheduling for Slurm versions >=22.05.
        # Can be removed if https://bugs.schedmd.com/show_bug.cgi?id=14298 is ever fixed (currently won't fix)
        os.environ.pop("SLURM_CPU_BIND", None)

        job_id, stderr = chcall(f"sbatch --parsable {filename}")
        os.unlink(filename)

        if len(stderr) > 0:
            logging.warning(f"Submitting batch job emitted warnings: {stderr}")

        return str(int(job_id))  # int() ensures coherent parsing

    def inner_handle_kill(self, *args: Any, **kwargs: Any) -> None:  # noqa ARG002 Unused method argument: `args`, kwargs
        for submit_thread in self.submit_threads:
            submit_thread.stop()

        # Jobs with a NOT_YET_SUBMITTED_STATE have not been submitted to the cluster yet
        scheduled_job_ids: list[int | str] = [
            job_id
            for job_id, job_state in self.jobs.items()
            if job_state != NOT_YET_SUBMITTED_STATE
        ]

        if len(scheduled_job_ids):
            # Array jobs (whose id looks like `<job_id>_<array_index>`) don't need to be signaled individually,
            # but can be canceled together using the job_id.
            unique_job_ids = set(map(lambda x: str(x).split("_")[0], scheduled_job_ids))
            job_id_string = " ".join(unique_job_ids)
            # Send SIGINT signal to running jobs instead of terminating the jobs right away. This way, the jobs can
            # react to the signal, safely shutdown and signal (cancel) jobs they possibly scheduled, recursively.
            _, stderr, _ = call(
                f"scancel --state=PENDING {job_id_string}; scancel -s SIGINT --state=RUNNING {job_id_string}; scancel --state=SUSPENDED {job_id_string}"
            )

            maybe_error_or_warning = (
                f"\nErrors and warnings (if all jobs were pending 'Invalid job id' errors are expected):\n{stderr}"
                if stderr
                else ""
            )
            logging.debug(
                f"Canceled slurm jobs {', '.join(unique_job_ids)}.{maybe_error_or_warning}"
            )

    def cleanup_submit_threads(self) -> None:
        self.submit_threads = [
            thread for thread in self.submit_threads if thread.is_alive()
        ]

    def inner_submit(
        self,
        cmdline: str,
        job_name: str | None = None,
        additional_setup_lines: list[str] | None = None,
        job_count: int | None = None,
    ) -> tuple[list[Future[str]], list[tuple[int, int]]]:
        """Starts a Slurm job that runs the specified shell command line."""
        if additional_setup_lines is None:
            additional_setup_lines = []

        # These place holders will be replaced by sbatch, see https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E
        # This variable needs to be kept in sync with the job_id_string variable in the
        # get_job_id_string function.
        job_id_string = "%j" if job_count is None else "%A_%a"
        log_path = self.format_log_file_path(self.cfut_dir, job_id_string)

        job_resources_lines = []
        if self.job_resources is not None:
            for resource, value in self.job_resources.items():
                job_resources_lines += [f"#SBATCH --{resource}={value}"]

        max_array_size = self.get_max_array_size()
        max_submit_jobs = self.get_max_submit_jobs()
        max_running_size = self.get_max_running_size()
        slurm_max_running_size_str = (
            f"%{max_running_size}" if max_running_size > 0 else ""
        )
        # Only ever submit at most max_submit_jobs and max_array_size jobs at once (but at least one).
        batch_size = max(min(max_array_size, max_submit_jobs), 1)

        scripts = []
        job_id_futures: list[Future[str]] = []
        ranges = []
        number_of_jobs = job_count if job_count is not None else 1
        for job_index_start in range(0, number_of_jobs, batch_size):
            # job_index_end is inclusive
            job_index_end = min(job_index_start + batch_size, number_of_jobs) - 1
            array_index_end = job_index_end - job_index_start

            job_array_line = ""
            if job_count is not None:
                job_array_line = (
                    f"#SBATCH --array=0-{array_index_end}{slurm_max_running_size_str}"
                )
            script_lines = (
                [
                    "#!/bin/sh",
                    f"#SBATCH --output={log_path}",
                    f'#SBATCH --job-name "{job_name}"',
                    job_array_line,
                ]
                + job_resources_lines
                + [
                    *additional_setup_lines,
                    f"srun {cmdline} {job_index_start}",
                ]
            )

            job_id_futures.append(Future())
            scripts.append("\n".join(script_lines))
            ranges.append((job_index_start, job_index_end + 1))

        job_sizes = [end - start for start, end in ranges]

        self.cleanup_submit_threads()

        submit_thread = _JobSubmitThread(
            scripts, job_sizes, job_id_futures, self.cfut_dir
        )
        self.submit_threads.append(submit_thread)
        submit_thread.start()

        return job_id_futures, ranges

    def check_job_state(
        self, job_id_with_index: str
    ) -> Literal["failed", "ignore", "completed"]:
        job_states = []

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call(f"scontrol show job {job_id_with_index}")

        if exit_code == 0:
            job_state_search = re.search("JobState=([a-zA-Z_]*)", stdout)
            if job_state_search:
                job_states = [job_state_search.group(1)]
            else:
                logging.error(f"Could not extract slurm job state? {stdout[0:10]}")
        else:
            stdout, _, exit_code = call(f"sacct -j {job_id_with_index} -o State -P")

            if exit_code == 0:
                job_states = stdout.split("\n")[1:]

        if len(job_states) == 0:
            logging.error(
                "Couldn't call scontrol nor sacct to determine job's status. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
            )
            return "ignore"

        def matches_states(slurm_states: list[str]) -> bool:
            return len(list(set(job_states) & set(slurm_states))) > 0

        if matches_states(SLURM_STATES["Failure"]):
            return "failed"
        elif matches_states(SLURM_STATES["Ignore"]):
            return "ignore"
        elif matches_states(SLURM_STATES["Unclear"]):
            logging.warning(
                f"The job state for {job_id_with_index} is {job_states}. It's unclear whether the job will recover. Will wait further"
            )
            return "ignore"
        elif matches_states(SLURM_STATES["Success"]):
            return "completed"
        else:
            logging.error(
                f"Unhandled slurm job state for job id {job_id_with_index}? {job_states}"
            )
            return "ignore"

    def investigate_failed_job(
        self, job_id_with_index: str
    ) -> tuple[str, type[RemoteException]] | None:
        # This function tries to find the reason for a failed job by first checking whether
        # the job run time exceeded the specified time limit. If that is not the case, it
        # checks whether the job used too much RAM. As a last resort, it checks the exit code
        # of the job: If the job was killed with signal 9, it's very likely due to some
        # resource limit.

        def parse_key_value_pairs(
            text: str, pair_delimiter: str, key_value_delimiter: str
        ) -> dict[str, str]:
            properties = {}
            for key_value_pair in text.split(pair_delimiter):
                if key_value_delimiter not in key_value_pair:
                    continue
                key, value = key_value_pair.split(key_value_delimiter, 1)
                properties[key.strip()] = value.strip()
            return properties

        # Call `scontrol show jobid=<job_id>` which should return some output including
        # key=value pairs, such as: "Reason=...", "TimeLimit=...", and "RunTime=..."
        stdout, stderr, exit_code = call(f"scontrol show jobid={job_id_with_index}")

        properties = None
        if exit_code == 0:
            # Parse stdout into a key-value object
            properties = parse_key_value_pairs(stdout, " ", "=")

            time_limit_investigation = self._investigate_time_limit(properties)
            if time_limit_investigation:
                return time_limit_investigation

        # Request gathered job information.
        for _ in range(10):
            stdout, _, exit_code = call(
                f"sacct -P --format=JobID,State,MaxRSS,ReqMem --unit K -j {job_id_with_index}"
            )
            stdout_lines = stdout.splitlines()

            if exit_code != 0:
                break
            elif len(stdout_lines) <= 1:
                time.sleep(0.2)
            else:
                # Parse stdout into a key-value object
                memory_limit_investigation = self._investigate_memory_consumption(
                    stdout
                )
                if memory_limit_investigation:
                    return memory_limit_investigation
                break

        if properties is None:
            return None

        return self._investigate_exit_code(properties)

    def _investigate_time_limit(
        self, properties: dict[str, str]
    ) -> tuple[str, type[RemoteTimeLimitException]] | None:
        reason = properties.get("Reason", None)
        if not reason:
            return None

        if reason != "TimeLimit":
            return None

        time_limit = properties.get("TimeLimit", None)
        run_time = properties.get("RunTime", None)
        time_limit_note = f"Time Limit: {time_limit} Run Time: {run_time}"

        reason = f"The job was probably terminated because it ran for too long ({time_limit_note})."
        return (reason, RemoteTimeLimitException)

    def _investigate_memory_consumption(
        self, stdout: str
    ) -> tuple[str, type[RemoteOutOfMemoryException]] | None:
        # Table Format:
        #
        #   JobID|State|MaxRSS|ReqMem
        #   91_0|FAILED||30720K
        #   91_0.batch|FAILED|248K|
        #   91_0.0|OUT_OF_MEMORY|164K|
        max_rss = 0
        req_mem = 0
        states = []
        linefilter = re.compile(r"^([^|]*)\|(\w*)\|(\d*)K?\|(\d*)K?$", re.MULTILINE)
        for job_id, state, rss, mem in linefilter.findall(stdout):
            max_rss = max(max_rss, int(rss or 0))
            req_mem = max(req_mem, int(mem or 0))
            states += [state]

        if "OUT_OF_MEMORY" in states:
            # Check if task plugin killed the job. This is the case if cgroup is used as TaskPlugin and
            # memory limits are enforced.
            reason = f"The job was terminated because it consumed too much memory (Requested: {req_mem / 1000} MB)."
        elif max_rss > req_mem:
            # Check if job accounting canceled the job. This is the case if JobAcctGatherParam=OverMemoryKill
            # is enabled.
            reason = f"The job was probably terminated because it consumed too much memory. Required {max_rss / 1000} MB but requeseted {req_mem / 1000} MB."
        else:
            return None
        return (reason, RemoteOutOfMemoryException)

    def _investigate_exit_code(
        self, properties: dict[str, str]
    ) -> tuple[str, type[RemoteResourceLimitException]] | None:
        if not properties.get("State", None):
            return None
        # For exit codes >128, subtract 128 to obtain the linux signal number which is SIGKILL (9) in this case
        if "exit code 137" not in properties["State"]:
            return None
        reason = (
            "The job was probably terminated because it consumed too much memory "
            "or ran for too long (at least, the exit code 137 suggests this). Please "
            "use the `seff` utility to inspect the failed job and its potential "
            "job siblings (in case of an array job) to doublecheck the memory "
            "consumption and run time."
        )
        return (reason, RemoteResourceLimitException)

    def get_pending_tasks(self) -> Iterable[str]:
        try:
            # Get the job ids (%i) of the active user (-u) which are pending (-t) and format
            # them one-per-line (-r) while excluding the header (-h).
            stdout, _ = chcall("squeue -u $(whoami) -t PENDING -r -h --format=%i")

            job_ids = set(stdout.split("\n"))
            return job_ids
        except Exception:
            logging.error(
                "Couldn't query pending jobs. Polling for finished jobs might be slow."
            )
            return []


class _JobSubmitThread(threading.Thread):
    def __init__(
        self,
        scripts: list[str],
        job_sizes: list[int],
        futures: list[Future[str]],
        cfut_dir: str,
    ):
        super().__init__()
        self._stop_event = threading.Event()
        self.scripts = scripts
        self.job_sizes = job_sizes
        self.futures = futures
        self.cfut_dir = cfut_dir

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        max_submit_jobs = SlurmExecutor.get_max_submit_jobs()

        for script, job_size, future in zip(self.scripts, self.job_sizes, self.futures):
            if self._stop_event.is_set():
                return

            while (
                SlurmExecutor.get_number_of_submitted_jobs() + job_size
                > max_submit_jobs
            ):
                # _stop_event.wait will wait for SLURM_QUEUE_CHECK_INTERVAL unless the event is signaled
                # in which case the thread was stopped
                self._stop_event.wait(SLURM_QUEUE_CHECK_INTERVAL)

                if self._stop_event.is_set():
                    return

            job_id = SlurmExecutor.submit_text(script, self.cfut_dir)
            future.set_result(job_id)
