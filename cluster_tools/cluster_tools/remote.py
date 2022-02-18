"""Tools for executing remote commands."""
import logging
import os
import sys
import traceback

from cluster_tools.schedulers.kube import KubernetesExecutor
from cluster_tools.schedulers.pbs import PBSExecutor
from cluster_tools.schedulers.slurm import SlurmExecutor
from cluster_tools.util import with_preliminary_postfix

from . import pickling


def get_executor_class(executor_key):
    return {
        "slurm": SlurmExecutor,
        "pbs": PBSExecutor,
        "kubernetes": KubernetesExecutor,
    }.get(executor_key)


def format_remote_exc():
    typ, value, tb = sys.exc_info()
    tb = tb.tb_next  # Remove root call to worker().
    return "".join(traceback.format_exception(typ, value, tb))


def get_custom_main_path(workerid, executor):
    custom_main_path = None
    main_meta_path = executor.get_main_meta_path(cfut_dir, workerid)
    if os.path.exists(main_meta_path):
        with open(main_meta_path, "r") as file:
            custom_main_path = file.read()
    return custom_main_path


def worker(executor, workerid, job_array_index, job_array_index_offset, cfut_dir):
    """Called to execute a job on a remote host."""

    if job_array_index is not None:
        workerid_with_idx = (
            worker_id + "_" + str(int(job_array_index_offset) + int(job_array_index))
        )
    else:
        workerid_with_idx = worker_id

    try:
        input_file_name = executor.format_infile_name(cfut_dir, workerid_with_idx)
        print("trying to read: ", input_file_name)
        print("working dir: ", os.getcwd())

        custom_main_path = get_custom_main_path(workerid, executor)
        with open(input_file_name, "rb") as f:
            unpickled_tuple = pickling.load(f, custom_main_path)
            if len(unpickled_tuple) == 4:
                fun, args, kwargs, meta_data = unpickled_tuple
                output_pickle_path = executor.format_outfile_name(
                    cfut_dir, workerid_with_idx
                )
            else:
                assert len(unpickled_tuple) == 5, "Unexpected encoding"
                fun, args, kwargs, meta_data, output_pickle_path = unpickled_tuple

        if type(fun) == str:
            with open(fun, "rb") as function_file:
                fun = pickling.load(function_file, custom_main_path)

        setup_logging(meta_data, executor, cfut_dir)

        logging.info(
            "Job computation started (jobid={}, workerid_with_idx={}).".format(
                executor.get_current_job_id(), workerid_with_idx
            )
        )
        result = True, fun(*args, **kwargs)
        logging.info("Job computation completed.")
        out = pickling.dumps(result)

    except Exception:
        print(traceback.format_exc())

        result = False, format_remote_exc()
        logging.warning("Job computation failed.")
        out = pickling.dumps(result)

    # The .preliminary postfix is added since the output can
    # contain a serialized exception. If that is the case,
    # the file should not be used as a checkpoint by users
    # of the clustertools. Therefore, the postfix is only
    # removed by the polling party (ClusterExecutor) after
    # the success case was recognized.
    destfile = with_preliminary_postfix(output_pickle_path)
    tempfile = str(destfile) + ".tmp"
    with open(tempfile, "wb") as f:
        f.write(out)
    logging.debug("Pickle file written to {}.".format(tempfile))
    os.rename(tempfile, destfile)
    logging.debug("Pickle file renamed to {}.".format(destfile))


def setup_logging(meta_data, executor, cfut_dir):
    if "logging_setup_fn" in meta_data:
        logging.debug("Using supplied logging_setup_fn to setup logging.")
        job_id_string = executor.get_job_id_string()
        # Leave the log file suffix so the caller can add their own suffix
        log_file_path = executor.format_log_file_path(
            cfut_dir, job_id_string, suffix=""
        )
        meta_data["logging_setup_fn"](log_file_path)
    else:
        logging_config = meta_data.get(
            "logging_config",
            {"level": logging.DEBUG, "format": "%(asctime)s %(levelname)s %(message)s"},
        )

        logging.debug(
            "Setting up logging.basicConfig (potentially overwriting logging configuration of the main script). Config: {}".format(
                logging_config
            )
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
