"""Tools for executing remote commands."""

import logging
import os
import sys
import traceback
from typing import Any

from cluster_tools._utils import pickling
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from cluster_tools.schedulers.kube import KubernetesExecutor
from cluster_tools.schedulers.pbs import PBSExecutor
from cluster_tools.schedulers.slurm import SlurmExecutor


def get_executor_class(executor_key: str) -> type[ClusterExecutor]:
    return {
        "slurm": SlurmExecutor,
        "pbs": PBSExecutor,
        "kubernetes": KubernetesExecutor,
    }[executor_key]


def format_remote_exc() -> str:
    typ, value, tb = sys.exc_info()
    if tb is not None:
        tb = tb.tb_next  # Remove root call to worker().
    return "".join(traceback.format_exception(typ, value, tb))


def get_custom_main_path(workerid: str, executor: type[ClusterExecutor]) -> str | None:
    custom_main_path = None
    main_meta_path = executor.get_main_meta_path(cfut_dir, workerid)
    if os.path.exists(main_meta_path):
        with open(main_meta_path, encoding="utf-8") as file:
            custom_main_path = file.read()
    return custom_main_path


def worker(
    executor: type[ClusterExecutor],
    workerid: str,
    job_array_index: int | None,
    job_array_index_offset: str,
    cfut_dir: str,
) -> None:
    """Called to execute a job on a remote host."""

    if job_array_index is not None:
        workerid_with_idx = (
            worker_id + "_" + str(int(job_array_index_offset) + job_array_index)
        )
    else:
        workerid_with_idx = worker_id

    try:
        input_file_name = executor.format_infile_name(cfut_dir, workerid_with_idx)
        logging.debug(f"Trying to read: {input_file_name} (working dir: {os.getcwd()}")

        custom_main_path = get_custom_main_path(workerid, executor)
        with open(input_file_name, "rb") as f:
            unpickled_tuple = pickling.load(f, custom_main_path)
            assert len(unpickled_tuple) == 5, "Unexpected encoding"
            fun_and_metadata, args, kwargs, output_key, output_store = unpickled_tuple

        if isinstance(fun_and_metadata, str):
            with open(fun_and_metadata, "rb") as function_file:
                fun, meta_data = pickling.load(function_file, custom_main_path)
        else:
            fun, meta_data = fun_and_metadata

        setup_logging(meta_data, executor, cfut_dir)

        logging.info(
            f"Job computation started (jobid={executor.get_current_job_id()}, workerid_with_idx={workerid_with_idx})."
        )
        result = fun(*args, **kwargs)
        logging.info("Job computation completed.")
        success, out = True, pickling.dumps((True, result))

    except Exception:
        logging.warning(f"Job computation failed with:\n\n{traceback.format_exc()}")
        success, out = False, pickling.dumps((False, format_remote_exc()))

    # The store keeps failed outputs apart from successful ones (which serve as
    # checkpoints). The polling ClusterExecutor picks up both.
    try:
        output_store.write(output_key, out, success=success)
    except Exception:
        if not success:
            raise
        # A failing store fails the job, since the result would not be checkpointed.
        logging.warning(f"Writing the output failed with:\n\n{traceback.format_exc()}")
        output_store.write(
            output_key, pickling.dumps((False, format_remote_exc())), success=False
        )
        success = False
    logging.debug(f"Output written to store (key={output_key}, success={success}).")


def setup_logging(
    meta_data: dict[str, Any], executor: type[ClusterExecutor], cfut_dir: str
) -> None:
    if "logging_setup_fn" in meta_data:
        logging.debug("Using supplied logging_setup_fn to setup logging.")
        job_id_string = executor.get_job_id_string()
        # Leave the log file suffix so the caller can add their own suffix
        log_file_path = executor.format_log_file_path(
            cfut_dir, job_id_string, suffix=""
        )
        meta_data["logging_setup_fn"](log_file_path)
    else:
        logging_config = meta_data.get("logging_config", dict())

        logging.debug(
            f"Setting up logging.basicConfig (potentially overwriting logging configuration of the main script). Config: {logging_config}"
        )

        # Call basicConfig which is necessary for the logging to work.
        logging.basicConfig(**logging_config)

        # It can happen that the pickled logger was already initialized. In this case,
        # the above basicConfig call was a noop. Therefore, we have to set the level explicitly.
        logger = logging.getLogger()
        if "level" in logging_config:
            logger.setLevel(logging_config["level"])


if __name__ == "__main__":
    executor_key = sys.argv[1]
    executor = get_executor_class(executor_key)
    worker_id = sys.argv[2]
    cfut_dir = sys.argv[3]
    job_array_index_offset = sys.argv[4] if len(sys.argv) > 4 else "0"
    job_array_index = executor.get_job_array_index()

    worker(executor, worker_id, job_array_index, job_array_index_offset, cfut_dir)
    # This is a workaround for the case that some subprocesses are still hanging around and are waited for.
    # If this point is reached, results were written to disk and we can "safely" shut down everything.
    sys.exit()
