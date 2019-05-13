from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import threading
import time
from . import slurm
from .remote import INFILE_FMT, OUTFILE_FMT
from .util import random_string, local_filename, call
from . import pickling
import logging
import importlib
import re

class RemoteException(Exception):
    def __init__(self, error, job_id):
        self.error = error
        self.job_id = job_id

    def __str__(self):
        return str(self.job_id) + "\n" + self.error.strip()

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


class FileWaitThread(threading.Thread):
    """A thread that polls the filesystem waiting for a list of files to
    be created. When a specified file is created, it invokes a callback.
    """

    def __init__(self, callback, interval=1):
        """The callable ``callback`` will be invoked with value
        associated with the filename of each file that is created.
        ``interval`` specifies the polling rate.
        """
        threading.Thread.__init__(self)
        self.callback = callback
        self.interval = interval
        self.waiting = {}
        self.lock = threading.Lock()
        self.shutdown = False

    def stop(self):
        """Stop the thread soon."""
        with self.lock:
            self.shutdown = True

    def wait(self, filename, value):
        """Adds a new filename (and its associated callback value) to
        the set of files being waited upon.
        """
        with self.lock:
            self.waiting[filename] = value

    def run(self):

        def handle_completed_job(job_id, filename, failed_early):
            self.callback(job_id, failed_early)
            del self.waiting[filename]

        while True:
            with self.lock:
                if self.shutdown:
                    return

                # Poll for each file.
                for filename in list(self.waiting):
                    job_id = self.waiting[filename]

                    if os.path.exists(filename):
                        # Check for output file as a fast indicator for job completion
                        handle_completed_job(job_id, filename, False)
                    else:
                        # If the output file was not found, we determine the job status so that
                        # we can recognize jobs which failed hard (in this case, they don't produce output files)
                        stdout, _, exit_code = call("scontrol show job {}".format(job_id))

                        # We have to re-check for the output file since this could be created in the mean time
                        if os.path.exists(filename):
                            handle_completed_job(job_id, filename, False)
                        else:
                            if exit_code != 0:
                                logging.error(
                                    "Couldn't call scontrol to determine job's status. {}. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
                                )
                            else:
                                job_state_search = re.search('JobState=([a-zA-Z_]*)', str(stdout))

                                if job_state_search:
                                    job_state = job_state_search.group(1)

                                    if job_state in SLURM_STATES["Failure"]:
                                        handle_completed_job(job_id, filename, True)
                                    elif job_state in SLURM_STATES["Ignore"]:
                                        # This job state can be ignored
                                        pass
                                    elif job_state in SLURM_STATES["Unclear"]:
                                        logging.warn("The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(job_id, job_state))
                                    elif job_state in SLURM_STATES["Success"]:
                                        logging.error(
                                            "Job state is completed, but {} couldn't be found.".format(
                                                filename
                                            )
                                        )
                                        handle_completed_job(job_id, filename, True)

            time.sleep(self.interval)


class SlurmExecutor(futures.Executor):
    """Futures executor for executing jobs on a Slurm cluster."""

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

        self.wait_thread = FileWaitThread(self._completion)
        self.wait_thread.start()

        self.meta_data = {}
        if "logging_config" in kwargs:
            self.meta_data["logging_config"] = kwargs["logging_config"]

    def _start(self, workerid, job_count=None):
        """Start a job with the given worker ID and return an ID
        identifying the new job. The job should run ``python -m
        cfut.remote <workerid>.
        """
        return slurm.submit(
            "{} -m cluster_tools.remote {}".format(sys.executable, workerid),
            job_resources=self.job_resources,
            job_name=self.job_name,
            additional_setup_lines=self.additional_setup_lines,
            job_count=job_count,
        )

    def _cleanup(self, jobid):
        """Given a job ID as returned by _start, perform any necessary
        cleanup after the job has finished.
        """
        if self.keep_logs:
            return

        outf = slurm.OUTFILE_FMT.format(str(jobid))
        try:
            os.unlink(outf)
        except OSError:
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
            # this circumstance because the whole slurm job was marked as failed.
            # Therefore, we don't try to deserialize pickling output.
            success = False
            result = "Job submission/execution failed. Please look into the log file at {}".format(
                slurm.OUTFILE_FMT.format(jobid)
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
                "submit() was invoked on a SlurmExecutor instance even though shutdown() was executed for that instance."
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

        jobid = self._start(workerid)

        if self.debug:
            print("job submitted: %i" % jobid, file=sys.stderr)

        # Thread will wait for it to finish.
        self.wait_thread.wait(OUTFILE_FMT % workerid, jobid)

        with self.jobs_lock:
            self.jobs[jobid] = (fut, workerid)

        fut.slurm_jobid = jobid
        return fut

    def map_to_futures(self, fun, allArgs):
        self.ensure_not_shutdown()
        allArgs = list(allArgs)

        futs = []
        workerid = random_string()

        get_workerid_with_index = lambda index: workerid + "_" + str(index)

        # Submit jobs eagerly
        for index, arg in enumerate(allArgs):
            fut = futures.Future()

            # Start the job.
            funcser = pickling.dumps((fun, [arg], {}, self.meta_data), True)
            infile_name = INFILE_FMT % get_workerid_with_index(index)

            with open(infile_name, "wb") as f:
                f.write(funcser)

            futs.append(fut)

        job_count = len(allArgs)
        jobid = self._start(workerid, job_count)
        get_jobid_with_index = lambda index: str(jobid) + "_" + str(index)

        if self.debug:
            print(
                "main job submitted: %i. consists of %i subjobs." % (jobid, job_count),
                file=sys.stderr,
            )

        with self.jobs_lock:
            for index, fut in enumerate(futs):
                jobid_with_index = get_jobid_with_index(index)
                # Thread will wait for it to finish.
                workerid_with_index = get_workerid_with_index(index)
                self.wait_thread.wait(
                    OUTFILE_FMT % workerid_with_index, jobid_with_index
                )

                fut.slurm_jobid = jobid
                fut.slurm_jobindex = index

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
                "The provided chunksize parameter is ignored by SlurmExecutor."
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


def get_existent_kwargs_subset(white_list, kwargs):
    new_kwargs = {}
    for arg_name in white_list:
        if arg_name in kwargs:
            new_kwargs[arg_name] = kwargs[arg_name]

    return new_kwargs


class WrappedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, **kwargs):
        white_list = ["max_workers", "mp_context", "initializer", "initargs"]
        new_kwargs = get_existent_kwargs_subset(white_list, kwargs)

        ProcessPoolExecutor.__init__(self, **new_kwargs)


    def map_unordered(self, func, args):

        futs = [self.submit(func, arg) for arg in args]

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator():
            for fut in futures.as_completed(futs):
                yield fut.result()

        return result_generator()


class SequentialExecutor(WrappedProcessPoolExecutor):
    def __init__(self, **kwargs):
        kwargs["max_workers"] = 1
        WrappedProcessPoolExecutor.__init__(self, **kwargs)


def get_executor(environment, **kwargs):
    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "multiprocessing":
        return WrappedProcessPoolExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
