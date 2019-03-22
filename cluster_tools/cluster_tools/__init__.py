from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import threading
import time
from . import slurm
from .remote import INFILE_FMT, OUTFILE_FMT
from .util import random_string, local_filename, chcall
import cloudpickle
import logging

class RemoteException(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return '\n' + self.error.strip()

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
        while True:
            with self.lock:
                if self.shutdown:
                    return

                # Poll for each file.
                for filename in list(self.waiting):
                    job_id = self.waiting[filename]
                    if os.path.exists(filename):
                        self.callback(job_id, False)
                        del job_id
                    else:
                        # Let's get the state for the job to check whether it failed
                        stdout = chcall('sacct -j {} -o State,ExitCode -P'.format(job_id))
                        if "FAILED" in str(stdout[0]):
                            self.callback(job_id, True)
                            del job_id

            time.sleep(self.interval)

class ClusterExecutor(futures.Executor):
    """An abstract base class for executors that run jobs on clusters.
    """
    def __init__(self, debug=False, keep_logs=False):
        os.makedirs(local_filename(), exist_ok=True)
        self.debug = debug

        self.jobs = {}
        self.job_outfiles = {}
        self.jobs_lock = threading.Lock()
        self.jobs_empty_cond = threading.Condition(self.jobs_lock)
        self.keep_logs = keep_logs

        self.wait_thread = FileWaitThread(self._completion)
        self.wait_thread.start()

    def _start(workerid):
        """Start a job with the given worker ID and return an ID
        identifying the new job. The job should run ``python -m
        cfut.remote <workerid>.
        """
        raise NotImplementedError()

    def _cleanup(jobid):
        """Given a job ID as returned by _start, perform any necessary
        cleanup after the job has finished.
        """

    def _completion(self, jobid, failed_early):
        """Called whenever a job finishes."""
        with self.jobs_lock:
            fut, workerid = self.jobs.pop(jobid)
            if not self.jobs:
                self.jobs_empty_cond.notify_all()
        if self.debug:
            print("job completed: %i" % jobid, file=sys.stderr)

        if failed_early:
            # If the code which should be executed on a node wasn't even
            # started (e.g., because python isn't installed or the cluster_tools
            # couldn't be found), no output was written to disk. We only noticed
            # this circumstance because the whole slurm job was marked as failed.
            # Therefore, we don't try to deserialize pickle output.
            success = False
            result = "Job submission/execution failed. Please look into the log file at {}".format(slurm.OUTFILE_FMT.format(jobid))
        else:
            with open(OUTFILE_FMT % workerid, 'rb') as f:
                outdata = f.read()
            success, result = cloudpickle.loads(outdata)

        if success:
            fut.set_result(result)
        else:
            fut.set_exception(RemoteException(result))

        # Clean up communication files.

        if os.path.exists(INFILE_FMT % workerid):
            os.unlink(INFILE_FMT % workerid)
        if os.path.exists(OUTFILE_FMT % workerid):
            os.unlink(OUTFILE_FMT % workerid)

        self._cleanup(jobid)

    def submit(self, fun, *args, additional_setup_lines=[], job_resources=None, **kwargs):
        """Submit a job to the pool."""
        fut = futures.Future()

        # Start the job.
        workerid = random_string()
        funcser = cloudpickle.dumps((fun, args, kwargs), True)
        with open(INFILE_FMT % workerid, 'wb') as f:
            f.write(funcser)
        jobid = self._start(workerid, additional_setup_lines, job_resources)

        if self.debug:
            print("job submitted: %i" % jobid, file=sys.stderr)

        # Thread will wait for it to finish.
        self.wait_thread.wait(OUTFILE_FMT % workerid, jobid)

        with self.jobs_lock:
            self.jobs[jobid] = (fut, workerid)

        fut.slurm_jobid = jobid
        return fut

    def shutdown(self, wait=True):
        """Close the pool."""
        if wait:
            with self.jobs_lock:
                if self.jobs:
                    self.jobs_empty_cond.wait()

        self.wait_thread.stop()
        self.wait_thread.join()

class SlurmExecutor(ClusterExecutor):
    """Futures executor for executing jobs on a Slurm cluster."""
    def _start(self, workerid, additional_setup_lines, job_resources):
        return slurm.submit(
            '{} -m cluster_tools.remote {}'.format(sys.executable, workerid),
            additional_setup_lines=additional_setup_lines,
            job_resources=job_resources
        )

    def _cleanup(self, jobid):
        if self.keep_logs:
            return

        outf = slurm.OUTFILE_FMT.format(str(jobid))
        try:
            os.unlink(outf)
        except OSError:
            pass

    def map(self, func, args, timeout=None, chunksize=None):
        if chunksize is not None:
            logging.warning("The provided chunksize parameter is ignored by SlurmExecutor.")

        start_time = time.time()

        with self:
            futs = []
            for arg in args:
                futs.append(self.submit(func, arg))
            for fut in futs:
                passed_time = time.time() - start_time
                remaining_timeout = None if timeout is None else timeout - passed_time
                yield fut.result(remaining_timeout)


class SequentialExecutor(ProcessPoolExecutor):
    def __init__(self, **kwargs):
        max_workers = 1
        return ProcessPoolExecutor.__init__(self, max_workers, **kwargs)


def get_executor(environment, *args, **kwargs):
    if environment == "slurm":
        return SlurmExecutor(*args, **kwargs)
    elif environment == "multiprocessing":
        return ProcessPoolExecutor(*args, **kwargs)
    elif environment == "sequential":
        return SequentialExecutor(*args, **kwargs)
