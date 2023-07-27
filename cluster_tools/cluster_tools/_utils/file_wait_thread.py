import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from cluster_tools.schedulers.cluster_executor import ClusterExecutor


class FileWaitThread(threading.Thread):
    """A thread that polls the filesystem waiting for a list of files to
    be created. When a specified file is created, it invokes a callback.
    """

    MAX_RETRY = 30

    def __init__(
        self,
        callback: Callable[[str, bool], None],
        executor: "ClusterExecutor",
        interval: int = 2,
    ):
        """The callable ``callback`` will be invoked with value
        associated with the filename of each file that is created.
        ``interval`` specifies the polling rate.
        """
        threading.Thread.__init__(self)
        self.callback = callback
        self.interval = interval
        self.waiting: Dict[str, str] = {}
        self.retryMap: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.shutdown = False
        self.executor = executor

    def stop(self) -> None:
        """Stop the thread soon."""
        with self.lock:
            self.shutdown = True

    def waitFor(self, filename: str, value: str) -> None:
        """Adds a new filename (and its associated callback value) to
        the set of files being waited upon.
        """
        with self.lock:
            self.waiting[filename] = value

    def run(self) -> None:
        def handle_completed_job(
            job_id: str, filename: str, failed_early: bool
        ) -> None:
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
                        status = self.executor.check_job_state(job_id)

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
