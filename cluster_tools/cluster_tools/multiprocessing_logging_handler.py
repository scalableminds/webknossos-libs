import functools
import logging
import multiprocessing
import sys
import threading
import traceback
from logging import getLogger
from logging.handlers import QueueHandler
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Any, List

# Inspired by https://stackoverflow.com/a/894284


class MultiProcessingHandler(logging.Handler):
    """This class wraps a logging handler and instantiates a multiprocessing queue.
    It asynchronously receives messages from the queue and emits them using the
    wrapped handler. The queue can be used by logging.QueueHandlers in other processes
    so that in a multiprocessing context all log messages are received by the main process.
    This is especially important if another start_method than `fork` is used as in that case
    the logging context is not copied to the subprocess but instead logging needs to be re-setup.
    """

    def __init__(self, name: str, wrapped_handler: logging.Handler):
        super().__init__()

        self.wrapped_handler = wrapped_handler

        self.setLevel(self.wrapped_handler.level)
        if self.wrapped_handler.formatter:
            self.setFormatter(self.wrapped_handler.formatter)
        self.filters = self.wrapped_handler.filters

        self.queue = multiprocessing.Manager().Queue(-1)
        self._is_closed = False
        # Use thread to asynchronously receive messages from the queue
        self._queue_thread = threading.Thread(target=self._receive, name=name)
        self._queue_thread.daemon = True
        self._queue_thread.start()

    def _receive(self) -> None:
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break

                # Avoid getting stuck if the handler was closed by setting a timeout
                record = self.queue.get(timeout=0.2)
                if record is not None:
                    self.wrapped_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):  # pylint: disable=try-except-raise
                raise
            # multiprocessing.managers.RemoteError pop up quite often.
            # It seems that they can be safely ignored, though.
            except (BrokenPipeError, EOFError, multiprocessing.managers.RemoteError):
                break
            except QueueEmpty:
                # This case is reached when the timeout in queue.get is hit. Pass, to
                # check whether the handler was closed.
                pass
            except Exception:
                traceback.print_exc(file=sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.wrapped_handler.emit(record)

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue_thread.join()

            self.wrapped_handler.close()
            super().close()


def _setup_logging_multiprocessing(queues: List[Queue], levels: List[int]) -> None:
    """Re-setup logging in a multiprocessing context (only needed if a start_method other than
    fork is used) by setting up QueueHandler loggers for each queue and level
    so that log messages are piped to the original loggers in the main process.
    """

    root_logger = getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    root_logger.setLevel(min(levels) if len(levels) else logging.DEBUG)
    for queue, level in zip(queues, levels):
        handler = QueueHandler(queue)
        handler.setLevel(level)
        root_logger.addHandler(handler)


def get_multiprocessing_logging_setup_fn() -> Any:
    root_logger = getLogger()

    queues = []
    levels = []
    for i, handler in enumerate(list(root_logger.handlers)):
        # Wrap logging handlers in MultiProcessingHandlers to make them work in a multiprocessing setup
        # when using start_methods other than fork, for example, spawn or forkserver
        if not isinstance(handler, MultiProcessingHandler):
            mp_handler = MultiProcessingHandler(
                f"multi-processing-handler-{i}", handler
            )

            root_logger.removeHandler(handler)
            root_logger.addHandler(mp_handler)
        else:
            mp_handler = handler

        queues.append(mp_handler.queue)
        levels.append(mp_handler.level)

    # Return a logging setup function that when called will setup QueueHandler loggers
    # reusing the queues of each wrapped MultiProcessingHandler. This way all log messages
    # are forwarded to the main process.
    return functools.partial(_setup_logging_multiprocessing, queues, levels)
