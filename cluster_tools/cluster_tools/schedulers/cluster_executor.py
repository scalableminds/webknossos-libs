import logging
import os
import signal
import sys
import threading
import time
from abc import abstractmethod
from concurrent import futures
from functools import partial
from typing import List, Optional, Tuple

from typing_extensions import Literal

from cluster_tools import pickling
from cluster_tools.pickling import file_path_to_absolute_module
from cluster_tools.tailf import Tail
from cluster_tools.util import (
    FileWaitThread,
    enrich_future_with_uncaught_warning,
    get_function_name,
    random_string,
    with_preliminary_postfix,
)


class RemoteException(Exception):
    def __init__(self, error, job_id):  # pylint: disable=super-init-not-called
        self.error = error
        self.job_id = job_id

    def __str__(self):
        return str(self.job_id) + "\n" + self.error.strip()


class ClusterExecutor(futures.Executor):
    """Futures executor for executing jobs on a cluster."""

    def __init__(
        self,
        debug=False,
        keep_logs=True,
        cfut_dir=None,
        job_resources=None,
        job_name=None,
        additional_setup_lines=None,
        **kwargs,
    ):
        """
        `kwargs` can be the following optional parameters:
            `logging_config`: An object containing a `level` key specifying the desired log level and/or a
                `format` key specifying the desired log format string. Cannot be specified together
                with `logging_setup_fn`.
            `logging_setup_fn`: A function setting up custom logging. The function will be called for
                remotely executed code (slurm, pbs) to re-setup logging. The function will be called with the
                default log file name. If the caller sets up file logging, this log file name should be adapted,
                for example, by adding a .mylog suffix. Cannot be specified together with `logging_config`.
        """
        self.debug = debug
        self.job_resources = job_resources
        self.additional_setup_lines = additional_setup_lines or []
        self.job_name = job_name
        self.was_requested_to_shutdown = False
        self.cfut_dir = (
            cfut_dir if cfut_dir is not None else os.getenv("CFUT_DIR", ".cfut")
        )
        self.files_to_clean_up = []

        logging.info(
            f"Instantiating ClusterExecutor. Log files are stored in {self.cfut_dir}"
        )

        # `jobs` maps from job id to (future, workerid, outfile_name, should_keep_output)
        # In case, job arrays are used: job id and workerid are in the format of
        # `job_id-job_index` and `workerid-job_index`.
        self.jobs = {}
        self.job_outfiles = {}
        self.jobs_lock = threading.Lock()
        self.jobs_empty_cond = threading.Condition(self.jobs_lock)
        self.keep_logs = keep_logs

        self.wait_thread = FileWaitThread(self._completion, self)
        self.wait_thread.start()

        os.makedirs(self.cfut_dir, exist_ok=True)

        signal.signal(signal.SIGINT, self.handle_kill)
        signal.signal(signal.SIGTERM, self.handle_kill)

        self.meta_data = {}
        assert not (
            "logging_config" in kwargs and "logging_setup_fn" in kwargs
        ), "Specify either logging_config OR logging_setup_fn but not both at once"
        if "logging_config" in kwargs:
            self.meta_data["logging_config"] = kwargs["logging_config"]
        if "logging_setup_fn" in kwargs:
            self.meta_data["logging_setup_fn"] = kwargs["logging_setup_fn"]

    @classmethod
    @abstractmethod
    def executor_key(cls):
        pass

    def handle_kill(self, _signum, _frame):
        self.wait_thread.stop()
        job_ids = ",".join(str(id) for id in self.jobs.keys())
        print(
            "A termination signal was registered. The following jobs are still running on the cluster:\n{}".format(
                job_ids
            )
        )
        sys.exit(130)

    @abstractmethod
    def check_for_crashed_job(
        self, job_id_with_index
    ) -> Literal["failed", "ignore", "completed"]:
        pass

    def _start(self, workerid, job_count=None, job_name=None):
        """Start job(s) with the given worker ID and return IDs
        identifying the new job(s). The job should run ``python -m
        cfut.remote <executorkey> <workerid>.
        """

        jobids_futures, job_index_ranges = self.inner_submit(
            f"{self.get_python_executable()} -m cluster_tools.remote {self.executor_key()} {workerid} {self.cfut_dir}",
            job_name=self.job_name if self.job_name is not None else job_name,
            additional_setup_lines=self.additional_setup_lines,
            job_count=job_count,
        )

        # Since not all jobs may be submitted immediately, cluster executors return
        # jobid futures in the inner_submit function. Also, since it may not be allowed
        # to submit all jobs at once, the jobs may be submitted in batches with each batch
        # containing a subset of all jobs identified by a separate jobid. The job_index_ranges
        # array of (start_index, end_index) tuples, indicates which of the job_count
        # jobs were submitted in each batch. start_index is inclusive whereas end_index
        # is not.
        return jobids_futures, job_index_ranges

    @abstractmethod
    def inner_submit(
        self,
        cmdline: str,
        job_name: Optional[str] = None,
        additional_setup_lines: Optional[List[str]] = None,
        job_count: Optional[int] = None,
    ) -> Tuple[List["futures.Future[str]"], List[Tuple[int, int]]]:
        pass

    def _cleanup(self, jobid):
        """Given a job ID as returned by _start, perform any necessary
        cleanup after the job has finished.
        """
        if self.keep_logs:
            return

        outf = self.format_log_file_path(self.cfut_dir, jobid)
        self.files_to_clean_up.append(outf)

    @staticmethod
    @abstractmethod
    def format_log_file_name(job_id_with_index, suffix=".stdout"):
        pass

    @classmethod
    def format_log_file_path(cls, cfut_dir, job_id_with_index, suffix=".stdout"):
        return os.path.join(
            cfut_dir, cls.format_log_file_name(job_id_with_index, suffix)
        )

    @classmethod
    @abstractmethod
    def get_job_id_string(self):
        pass

    @staticmethod
    def get_temp_file_path(cfut_dir, file_name):
        return os.path.join(cfut_dir, file_name)

    @staticmethod
    def format_infile_name(cfut_dir, job_id):
        return os.path.join(cfut_dir, "cfut.in.%s.pickle" % job_id)

    @staticmethod
    def format_outfile_name(cfut_dir, job_id):
        return os.path.join(cfut_dir, "cfut.out.%s.pickle" % job_id)

    def get_python_executable(self):
        return sys.executable

    def _completion(self, jobid, failed_early):
        """Called whenever a job finishes."""
        with self.jobs_lock:
            job_info = self.jobs.pop(jobid)
            if len(job_info) == 4:
                fut, workerid, outfile_name, should_keep_output = job_info
            else:
                # Backwards compatibility
                fut, workerid = job_info
                should_keep_output = False
                outfile_name = self.format_outfile_name(self.cfut_dir, workerid)

            if not self.jobs:
                self.jobs_empty_cond.notify_all()
        if self.debug:
            print("job completed: {}".format(jobid), file=sys.stderr)

        preliminary_outfile_name = with_preliminary_postfix(outfile_name)
        if failed_early:
            # If the code which should be executed on a node wasn't even
            # started (e.g., because python isn't installed or the cluster_tools
            # couldn't be found), no output was written to disk. We only noticed
            # this circumstance because the whole job was marked as failed.
            # Therefore, we don't try to deserialize pickling output.
            success = False
            result = "Job submission/execution failed. Please look into the log file at {}".format(
                self.format_log_file_path(self.cfut_dir, jobid)
            )
        else:
            with open(preliminary_outfile_name, "rb") as f:
                outdata = f.read()
            success, result = pickling.loads(outdata)

        if success:
            # Remove the .preliminary postfix since the job was finished
            # successfully. # Therefore, the result can be used as a checkpoint
            # by users of the clustertools.
            os.rename(preliminary_outfile_name, outfile_name)
            logging.debug("Pickle file renamed to {}.".format(outfile_name))

            fut.set_result(result)
        else:
            fut.set_exception(RemoteException(result, jobid))

        # Clean up communication files.

        infile_name = self.format_infile_name(self.cfut_dir, workerid)
        self.files_to_clean_up.append(infile_name)
        if not should_keep_output:
            self.files_to_clean_up.append(outfile_name)

        self._cleanup(jobid)

    def ensure_not_shutdown(self):
        if self.was_requested_to_shutdown:
            raise RuntimeError(
                "submit() was invoked on a ClusterExecutor instance even though shutdown() was executed for that instance."
            )

    def create_enriched_future(self):
        fut = futures.Future()
        enrich_future_with_uncaught_warning(fut)
        return fut

    def submit(self, fun, *args, **kwargs):
        """
        Submit a job to the pool.
        kwargs may contain __cfut_options which currently should look like:
        {
            "output_pickle_path": str
        }
        output_pickle_path defines where the pickled result should be stored.
        That file will not be removed after the job has finished.
        """
        fut = self.create_enriched_future()
        workerid = random_string()

        should_keep_output = False
        if "__cfut_options" in kwargs:
            should_keep_output = True
            output_pickle_path = kwargs["__cfut_options"]["output_pickle_path"]
            del kwargs["__cfut_options"]
        else:
            output_pickle_path = self.format_outfile_name(self.cfut_dir, workerid)

        self.ensure_not_shutdown()

        # Start the job.
        serialized_function_info = pickling.dumps(
            (fun, args, kwargs, self.meta_data, output_pickle_path)
        )
        with open(self.format_infile_name(self.cfut_dir, workerid), "wb") as f:
            f.write(serialized_function_info)

        self.store_main_path_to_meta_file(workerid)

        preliminary_output_pickle_path = with_preliminary_postfix(output_pickle_path)
        if os.path.exists(preliminary_output_pickle_path):
            logging.warning(
                f"Deleting stale output file at {preliminary_output_pickle_path}..."
            )
            os.unlink(preliminary_output_pickle_path)

        job_name = get_function_name(fun)
        jobids_futures, _ = self._start(workerid, job_name=job_name)
        # Only a single job was submitted
        jobid = jobids_futures[0].result()

        if self.debug:
            print(f"job submitted: {jobid}", file=sys.stderr)

        # Thread will wait for it to finish.
        self.wait_thread.waitFor(preliminary_output_pickle_path, jobid)

        with self.jobs_lock:
            self.jobs[jobid] = (fut, workerid, output_pickle_path, should_keep_output)

        fut.cluster_jobid = jobid
        return fut

    @classmethod
    def get_workerid_with_index(cls, workerid, index):
        return workerid + "_" + str(index)

    @classmethod
    def get_jobid_with_index(cls, jobid, index):
        return str(jobid) + "_" + str(index)

    def get_function_pickle_path(self, workerid):
        return self.format_infile_name(
            self.cfut_dir, self.get_workerid_with_index(workerid, "function")
        )

    @staticmethod
    def get_main_meta_path(cfut_dir, workerid):
        return os.path.join(cfut_dir, f"cfut.main_path.{workerid}.txt")

    def store_main_path_to_meta_file(self, workerid):
        with open(self.get_main_meta_path(self.cfut_dir, workerid), "w") as file:
            file.write(file_path_to_absolute_module(sys.argv[0]))

    def map_to_futures(self, fun, allArgs, output_pickle_path_getter=None):
        self.ensure_not_shutdown()
        allArgs = list(allArgs)
        if len(allArgs) == 0:
            return []

        should_keep_output = output_pickle_path_getter is not None

        futs_with_output_paths = []
        workerid = random_string()

        pickled_function_path = self.get_function_pickle_path(workerid)
        self.files_to_clean_up.append(pickled_function_path)
        with open(pickled_function_path, "wb") as file:
            pickling.dump(fun, file)
        self.store_main_path_to_meta_file(workerid)

        for index, arg in enumerate(allArgs):
            fut = self.create_enriched_future()
            workerid_with_index = self.get_workerid_with_index(workerid, index)

            if output_pickle_path_getter is None:
                output_pickle_path = self.format_outfile_name(
                    self.cfut_dir, workerid_with_index
                )
            else:
                output_pickle_path = output_pickle_path_getter(arg)

            preliminary_output_pickle_path = with_preliminary_postfix(
                output_pickle_path
            )
            if os.path.exists(preliminary_output_pickle_path):
                logging.warning(
                    f"Deleting stale output file at {preliminary_output_pickle_path}..."
                )
                os.unlink(preliminary_output_pickle_path)

            serialized_function_info = pickling.dumps(
                (pickled_function_path, [arg], {}, self.meta_data, output_pickle_path)
            )
            infile_name = self.format_infile_name(self.cfut_dir, workerid_with_index)

            with open(infile_name, "wb") as f:
                f.write(serialized_function_info)

            futs_with_output_paths.append((fut, output_pickle_path))

        with self.jobs_lock:
            # Use a separate loop to avoid having to acquire the jobs_lock many times
            # or for the full duration of the above loop
            for index in range(len(futs_with_output_paths)):
                workerid_with_index = self.get_workerid_with_index(workerid, index)
                # Register the job in the jobs array, although the jobid is not known yet.
                # Otherwise it might happen that self.jobs becomes empty, but some of the jobs were
                # not even submitted yet.
                self.jobs[workerid_with_index] = "pending"

        job_count = len(allArgs)
        job_name = get_function_name(fun)
        jobids_futures, job_index_ranges = self._start(workerid, job_count, job_name)

        number_of_batches = len(jobids_futures)
        for batch_index, (jobid_future, (job_index_start, job_index_end)) in enumerate(
            zip(jobids_futures, job_index_ranges)
        ):
            jobid_future.add_done_callback(
                partial(
                    self.register_jobs,
                    futs_with_output_paths[job_index_start:job_index_end],
                    workerid,
                    should_keep_output,
                    job_index_start,
                    f"{batch_index + 1}/{number_of_batches}",
                )
            )

        return [fut for (fut, _) in futs_with_output_paths]

    def register_jobs(
        self,
        futs_with_output_paths,
        workerid,
        should_keep_output,
        job_index_offset,
        batch_description,
        jobid_future,
    ):
        jobid = jobid_future.result()
        if self.debug:

            print(
                "Submitted array job {} with JobId {} and {} subjobs.".format(
                    batch_description, jobid, len(futs_with_output_paths)
                ),
                file=sys.stderr,
            )

        for array_index, (fut, output_path) in enumerate(futs_with_output_paths):
            jobid_with_index = self.get_jobid_with_index(jobid, array_index)

            # Thread will wait for it to finish.
            self.wait_thread.waitFor(
                with_preliminary_postfix(output_path), jobid_with_index
            )

            fut.cluster_jobid = jobid
            fut.cluster_jobindex = array_index

            job_index = job_index_offset + array_index
            workerid_with_index = self.get_workerid_with_index(workerid, job_index)
            # Remove the pending jobs entry and add the correct one
            with self.jobs_lock:
                del self.jobs[workerid_with_index]
                self.jobs[jobid_with_index] = (
                    fut,
                    workerid_with_index,
                    output_path,
                    should_keep_output,
                )

    def shutdown(self, wait=True):
        """Close the pool."""
        self.was_requested_to_shutdown = True
        if wait:
            with self.jobs_lock:
                if self.jobs:
                    self.jobs_empty_cond.wait()

        self.wait_thread.stop()
        self.wait_thread.join()

        for file_to_clean_up in self.files_to_clean_up:
            try:
                os.unlink(file_to_clean_up)
            except OSError:
                pass
        self.files_to_clean_up = []

    def map(self, func, args, timeout=None, chunksize=None):
        if chunksize is not None:
            logging.warning(
                "The provided chunksize parameter is ignored by ClusterExecutor."
            )

        start_time = time.time()

        futs = self.map_to_futures(func, args)

        # Return a separate generator as an iterator to avoid that the
        # map() method itself becomes a generator (due to the usage of yield).
        # If map() was a generator, the submit() calls would be invoked
        # lazily which can lead to a shutdown of the executor before
        # the submit calls are performed.
        def result_generator():
            for fut in futs:
                passed_time = time.time() - start_time
                remaining_timeout = None if timeout is None else timeout - passed_time
                yield fut.result(remaining_timeout)

        return result_generator()

    def map_unordered(self, func, args):
        futs = self.map_to_futures(func, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily.
        def result_generator():
            for fut in futures.as_completed(futs):
                yield fut.result()

        return result_generator()

    def forward_log(self, fut):
        """
        Takes a future from which the log file is forwarded to the active
        process. This method blocks as long as the future is not done.
        """

        log_path = self.format_log_file_path(self.cfut_dir, fut.cluster_jobid)
        # Don't use a logger instance here, since the child process
        # probably already used a logger.
        log_callback = lambda s: sys.stdout.write(f"(jid={fut.cluster_jobid}) {s}")
        tailer = Tail(log_path, log_callback)
        fut.add_done_callback(lambda _: tailer.cancel())

        # Poll until the log file exists
        while not (os.path.exists(log_path) or tailer.is_cancelled):
            time.sleep(2)

        # Log the output of the log file until future is resolved
        # by the done_callback we attached earlier.
        tailer.follow(2)
        return fut.result()

    @abstractmethod
    def get_pending_tasks(self):
        pass
