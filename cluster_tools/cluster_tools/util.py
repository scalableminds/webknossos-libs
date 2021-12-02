import logging
import os
import random
import string
import subprocess
import threading
import time


def local_filename(filename=""):
    return os.path.join(os.getenv("CFUT_DIR", ".cfut"), filename)


# Instantiate a dedicate generator to avoid being dependent on
# the global seed which some external code might have set.
random_generator = random.Random()


def random_string(length=32, chars=(string.ascii_letters + string.digits)):
    return "".join(random_generator.choice(chars) for i in range(length))


def call(command, stdin=None):
    """Invokes a shell command as a subprocess, optionally with some
    data sent to the standard input. Returns the standard output data,
    the standard error, and the return code.
    """
    if stdin is not None:
        stdin_flag = subprocess.PIPE
    else:
        stdin_flag = None
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=stdin_flag,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(stdin)
    return stdout, stderr, proc.returncode


class CommandError(Exception):
    """Raised when a shell command exits abnormally."""

    def __init__(self, command, code, stderr):  # pylint: disable=super-init-not-called
        self.command = command
        self.code = code
        self.stderr = stderr

    def __str__(self):
        return "%s exited with status %i: %s" % (
            repr(self.command),
            self.code,
            repr(self.stderr),
        )


def chcall(command, stdin=None):
    """Like ``call`` but raises an exception when the return code is
    nonzero. Only returns the stdout and stderr data.
    """
    stdout, stderr, code = call(command, stdin)
    if code != 0:
        raise CommandError(command, code, stderr)
    return stdout, stderr


def warn_after(job, seconds):
    """
    Use as decorator to warn when a function is taking longer than {seconds} seconds.
    """

    def outer(fn):
        def inner(*args, **kwargs):
            exceeded_timeout = [False]
            start_time = time.time()

            def warn_function():
                logging.warning(
                    "Function {} is taking suspiciously long (longer than {} seconds)".format(
                        job, seconds
                    )
                )
                exceeded_timeout[0] = True

            timer = threading.Timer(seconds, warn_function)
            timer.start()

            try:
                result = fn(*args, **kwargs)
                if exceeded_timeout[0]:
                    end_time = time.time()
                    logging.warning(
                        "Function {} succeeded after all (took {} seconds)".format(
                            job, int(end_time - start_time)
                        )
                    )
            finally:
                timer.cancel()
            return result

        return inner

    return outer


class FileWaitThread(threading.Thread):
    """A thread that polls the filesystem waiting for a list of files to
    be created. When a specified file is created, it invokes a callback.
    """

    MAX_RETRY = 30

    def __init__(self, callback, executor, interval=2):
        """The callable ``callback`` will be invoked with value
        associated with the filename of each file that is created.
        ``interval`` specifies the polling rate.
        """
        threading.Thread.__init__(self)
        self.callback = callback
        self.interval = interval
        self.waiting = {}
        self.retryMap = {}
        self.lock = threading.Lock()
        self.shutdown = False
        self.executor = executor

    def stop(self):
        """Stop the thread soon."""
        with self.lock:
            self.shutdown = True

    def waitFor(self, filename, value):
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

                pending_tasks = self.executor.get_pending_tasks()

                # Poll for each file.
                for filename in list(self.waiting):
                    job_id = self.waiting[filename]
                    if job_id in pending_tasks:
                        # Don't check status of pending tasks, since this
                        # can vastly slow down the polling.
                        continue

                    if os.path.exists(filename):
                        # Check for output file as a fast indicator for job completion
                        handle_completed_job(job_id, filename, False)
                    elif self.executor is not None:
                        status = self.executor.check_for_crashed_job(job_id)

                        # We have to re-check for the output file since this could be created in the mean time
                        if os.path.exists(filename):
                            handle_completed_job(job_id, filename, False)
                        else:
                            if status == "completed":
                                self.retryMap[filename] = self.retryMap.get(filename, 0)
                                self.retryMap[filename] += 1

                                if self.retryMap[filename] <= FileWaitThread.MAX_RETRY:
                                    # Retry by looping again
                                    logging.warning(
                                        "Job state is completed, but {} couldn't be found. Retrying {}/{}".format(
                                            filename,
                                            self.retryMap[filename],
                                            FileWaitThread.MAX_RETRY,
                                        )
                                    )
                                else:
                                    logging.error(
                                        "Job state is completed, but {} couldn't be found.".format(
                                            filename
                                        )
                                    )
                                    handle_completed_job(job_id, filename, True)

                            elif status == "failed":
                                handle_completed_job(job_id, filename, True)
                            elif status == "ignore":
                                pass
            time.sleep(self.interval)


def get_function_name(fun):
    # When using functools.partial, __name__ does not exist
    try:
        return fun.__name__ if hasattr(fun, "__name__") else fun.func.__name__
    except Exception:
        return "<unknown function>"


def enrich_future_with_uncaught_warning(f):
    def warn_on_exception(future):
        maybe_exception = future.exception()
        if maybe_exception is not None:
            logging.error(
                "A future crashed with an exception: {}. Future: {}".format(
                    maybe_exception, future
                )
            )

    if not hasattr(f, "is_wrapped_by_cluster_tools"):
        f.is_wrapped_by_cluster_tools = True
        f.add_done_callback(warn_on_exception)


def with_preliminary_postfix(name):
    return f"{name}.preliminary"
