from concurrent import futures
import os
from cluster_tools.util import random_string, local_filename, FileWaitThread
import threading
import signal
import sys
from cluster_tools import pickling
import time
from abc import ABC, abstractmethod
from cluster_tools.file_formatters import INFILE_FMT, OUTFILE_FMT
import logging
from typing import Union

class RemoteException(Exception):
    def __init__(self, error, job_id):
        self.error = error
        self.job_id = job_id

    def __str__(self):
        return str(self.job_id) + "\n" + self.error.strip()


class ClusterExecutor(futures.Executor):
    """Futures executor for executing jobs on a cluster."""

    def __init__(
        self,
        debug=False,
        keep_logs=False,
        job_resources=None,
        job_name=None,
        additional_setup_lines=[],
        **kwargs
    ):
        os.makedirs(local_filename(), exist_ok=True)
        self.debug = debug
        self.job_resources = job_resources
        self.additional_setup_lines = additional_setup_lines
        self.job_name = job_name
        self.was_requested_to_shutdown = False

        # `jobs` maps from job id to future and workerid
        # In case, job arrays are used: job id and workerid are in the format of
        # `job_id-job_index` and `workerid-job_index`.
        self.jobs = {}
        self.job_outfiles = {}
        self.jobs_lock = threading.Lock()
        self.jobs_empty_cond = threading.Condition(self.jobs_lock)
        self.keep_logs = keep_logs

        self.wait_thread = FileWaitThread(self._completion, self)
        self.wait_thread.start()

        signal.signal(signal.SIGINT, self.handle_kill)
        signal.signal(signal.SIGTERM, self.handle_kill)

        self.meta_data = {}
        if "logging_config" in kwargs:
            self.meta_data["logging_config"] = kwargs["logging_config"]


    def handle_kill(self,signum, frame):
      self.wait_thread.stop()
      job_ids = ",".join(str(id) for id in self.jobs.keys())
      print("A termination signal was registered. The following jobs are still running on the cluster:\n{}".format(job_ids))
      sys.exit(130)


    @abstractmethod
    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:
        pass


    def _start(self, workerid, job_count=None, job_name=None):
        """Start a job with the given worker ID and return an ID
        identifying the new job. The job should run ``python -m
        cfut.remote <workerid>.
        """
        return self.inner_submit(
            "{} -m cluster_tools.remote {}".format(sys.executable, workerid),
            job_name=self.job_name if self.job_name is not None else job_name,
            additional_setup_lines=self.additional_setup_lines,
            job_count=job_count,
        )

    @abstractmethod
    def inner_submit(self, *args, **kwargs):
        pass

    def _cleanup(self, jobid):
        """Given a job ID as returned by _start, perform any necessary
        cleanup after the job has finished.
        """
        if self.keep_logs:
            return

        outf = self.format_log_file_name(jobid)
        try:
            os.unlink(outf)
        except OSError:
            pass

    @abstractmethod
    def format_log_file_name(self, jobid):
        pass


    def _completion(self, jobid, failed_early):
        """Called whenever a job finishes."""
        with self.jobs_lock:
            fut, workerid = self.jobs.pop(jobid)
            if not self.jobs:
                self.jobs_empty_cond.notify_all()
        if self.debug:
            print("job completed: {}".format(jobid), file=sys.stderr)

        if failed_early:
            # If the code which should be executed on a node wasn't even
            # started (e.g., because python isn't installed or the cluster_tools
            # couldn't be found), no output was written to disk. We only noticed
            # this circumstance because the whole job was marked as failed.
            # Therefore, we don't try to deserialize pickling output.
            success = False
            result = "Job submission/execution failed. Please look into the log file at {}".format(
                self.format_log_file_name(jobid)
            )
        else:
            with open(OUTFILE_FMT % workerid, "rb") as f:
                outdata = f.read()
            success, result = pickling.loads(outdata)

        if success:
            fut.set_result(result)
        else:
            fut.set_exception(RemoteException(result, jobid))

        # Clean up communication files.

        if os.path.exists(INFILE_FMT % workerid):
            os.unlink(INFILE_FMT % workerid)
        if os.path.exists(OUTFILE_FMT % workerid):
            os.unlink(OUTFILE_FMT % workerid)

        self._cleanup(jobid)

    def ensure_not_shutdown(self):
        if self.was_requested_to_shutdown:
            raise RuntimeError(
                "submit() was invoked on a ClusterExecutor instance even though shutdown() was executed for that instance."
            )

    def submit(self, fun, *args, **kwargs):
        """Submit a job to the pool."""
        fut = futures.Future()

        self.ensure_not_shutdown()

        # Start the job.
        workerid = random_string()

        funcser = pickling.dumps((fun, args, kwargs, self.meta_data), True)
        with open(INFILE_FMT % workerid, "wb") as f:
            f.write(funcser)

        job_name = fun.__name__
        jobid = self._start(workerid, job_name=job_name)

        if self.debug:
            print("job submitted: %i" % jobid, file=sys.stderr)

        # Thread will wait for it to finish.
        self.wait_thread.waitFor(OUTFILE_FMT % workerid, jobid)

        with self.jobs_lock:
            self.jobs[jobid] = (fut, workerid)

        fut.cluster_jobid = jobid
        return fut

    def get_workerid_with_index(self, workerid, index):
        return workerid + "_" + str(index)

    def get_jobid_with_index(self, jobid, index):
        return str(jobid) + "_" + str(index)

    def map_to_futures(self, fun, allArgs):
        self.ensure_not_shutdown()
        allArgs = list(allArgs)

        futs = []
        workerid = random_string()

        # Submit jobs eagerly
        for index, arg in enumerate(allArgs):
            fut = futures.Future()

            # Start the job.
            funcser = pickling.dumps((fun, [arg], {}, self.meta_data), True)
            infile_name = INFILE_FMT % self.get_workerid_with_index(workerid, index)

            with open(infile_name, "wb") as f:
                f.write(funcser)

            futs.append(fut)

        job_count = len(allArgs)
        job_name = fun.__name__
        jobid = self._start(workerid, job_count, job_name)
        

        if self.debug:
            print(
                "main job submitted: %i. consists of %i subjobs." % (jobid, job_count),
                file=sys.stderr,
            )

        with self.jobs_lock:
            for index, fut in enumerate(futs):
                jobid_with_index = self.get_jobid_with_index(jobid, index)
                # Thread will wait for it to finish.
                workerid_with_index = self.get_workerid_with_index(workerid, index)
                self.wait_thread.waitFor(
                    OUTFILE_FMT % workerid_with_index, jobid_with_index
                )

                fut.cluster_jobid = jobid
                fut.cluster_jobindex = index

                self.jobs[jobid_with_index] = (fut, workerid_with_index)

        return futs

    def shutdown(self, wait=True):
        """Close the pool."""
        self.was_requested_to_shutdown = True
        if wait:
            with self.jobs_lock:
                if self.jobs:
                    self.jobs_empty_cond.wait()

        self.wait_thread.stop()
        self.wait_thread.join()


    def map(self, func, args, timeout=None, chunksize=None):
        if chunksize is not None:
            logging.warning(
                "The provided chunksize parameter is ignored by ClusterExecutor."
            )

        start_time = time.time()

        futs = self.map_to_futures(func, args)
        results = []

        # Return a separate generator as iterator to avoid that the
        # map() method itself becomes a generator.
        # If map() was a generator, the submit() calls would be invoked
        # lazily which can lead to a shutdown of the executor before
        # the submit calls are performed.
        def result_generator():
            for fut in futs:
                passed_time = time.time() - start_time
                remaining_timeout = (
                    None if timeout is None else timeout - passed_time
                )
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