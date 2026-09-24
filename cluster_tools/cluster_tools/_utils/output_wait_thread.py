import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cluster_tools.output_store import OutputStore
    from cluster_tools.schedulers.cluster_executor import ClusterExecutor


class OutputWaitThread(threading.Thread):
    """A thread that polls the executor's output store waiting for a set of
    output keys to be written. When a key is written, it invokes a callback.
    """

    MAX_RETRY = 30

    def __init__(
        self,
        callback: Callable[[str, bool], None],
        executor: "ClusterExecutor",
        interval: int = 2,
    ):
        """The callable ``callback`` will be invoked with the value
        associated with each key that is written.
        ``interval`` specifies the polling rate.
        """
        threading.Thread.__init__(self)
        self.callback = callback
        self.interval = interval
        self.waiting: dict[str, str] = {}
        self.retryMap: dict[str, int] = {}
        self.lock = threading.Lock()
        self.shutdown = False
        self.executor = executor

    def stop(self) -> None:
        """Stop the thread soon."""
        with self.lock:
            self.shutdown = True

    def waitFor(self, key: str, value: str) -> None:
        """Adds a new output key (and its associated callback value) to
        the set of keys being waited upon.
        """
        with self.lock:
            self.waiting[key] = value

    @staticmethod
    def _poll_single(store: "OutputStore", key: str) -> set[str]:
        try:
            return store.poll([key])
        except Exception:
            logging.warning(
                f"Polling the output store for {key} failed.", exc_info=True
            )
            return set()

    def run(self) -> None:
        def handle_completed_job(job_id: str, key: str, failed_early: bool) -> None:
            # The callback must not raise, since that would stop this thread and
            # leave all other jobs pending.
            try:
                self.callback(job_id, failed_early)
            except Exception:
                logging.exception(f"Handling the completion of job {job_id} failed.")
            del self.waiting[key]

        while True:
            with self.lock:
                if self.shutdown:
                    return

                pending_tasks = self.executor.get_pending_tasks()
                store = self.executor.output_store

                # Don't check status of pending tasks, since this
                # can vastly slow down the polling.
                keys_to_check = [
                    key
                    for key, job_id in self.waiting.items()
                    if job_id not in pending_tasks
                ]
                # Check for outputs in one batch as a fast indicator for job completion.
                # A store can be remote, so a failing poll must not kill this thread;
                # it is retried on the next iteration.
                try:
                    written_keys = store.poll(keys_to_check)
                except Exception:
                    logging.warning(
                        "Polling the output store failed, retrying.", exc_info=True
                    )
                    written_keys = set()

                for key in keys_to_check:
                    job_id = self.waiting[key]

                    if key in written_keys:
                        handle_completed_job(job_id, key, False)
                    elif self.executor is not None:
                        status = self.executor.check_job_state(job_id)

                        # We have to re-check for the output since this could be written in the mean time
                        if key in self._poll_single(store, key):
                            handle_completed_job(job_id, key, False)
                        else:
                            if status == "completed":
                                self.retryMap[key] = self.retryMap.get(key, 0)
                                self.retryMap[key] += 1

                                if self.retryMap[key] <= OutputWaitThread.MAX_RETRY:
                                    # Retry by looping again
                                    logging.warning(
                                        f"Job state is completed, but output {key} couldn't be found. Retrying {self.retryMap[key]}/{OutputWaitThread.MAX_RETRY}"
                                    )
                                else:
                                    logging.error(
                                        f"Job state is completed, but output {key} couldn't be found."
                                    )
                                    handle_completed_job(job_id, key, True)

                            elif status == "failed":
                                handle_completed_job(job_id, key, True)
                            elif status == "ignore":
                                pass
            time.sleep(self.interval)
