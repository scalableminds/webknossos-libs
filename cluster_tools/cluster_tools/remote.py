"""Tools for executing remote commands."""
import sys
import os
import traceback
from . import pickling
import logging
from cluster_tools.schedulers.slurm import SlurmExecutor
from cluster_tools.schedulers.pbs import PBSExecutor

def get_executor_class():
    for executor in [SlurmExecutor, PBSExecutor]:
        if executor.get_current_job_id() is not None:
            return executor

def format_remote_exc():
    typ, value, tb = sys.exc_info()
    tb = tb.tb_next  # Remove root call to worker().
    return "".join(traceback.format_exception(typ, value, tb))


def get_custom_main_path(workerid):
    custom_main_path = None
    main_meta_path = get_executor_class().get_main_meta_path(cfut_dir, workerid)
    if os.path.exists(main_meta_path):
        with open(main_meta_path, "r") as file:
            custom_main_path = file.read()
    return custom_main_path

def worker(workerid, job_array_index, cfut_dir):
    """Called to execute a job on a remote host."""

    workerid_with_idx = worker_id + "_" + job_array_index if job_array_index is not None else workerid

    executor = get_executor_class()
    try:
        input_file_name = executor.format_infile_name(cfut_dir, workerid_with_idx)
        print("trying to read: ", input_file_name)
        print("working dir: ", os.getcwd())

        custom_main_path = get_custom_main_path(workerid)
        with open(input_file_name, "rb") as f:
            unpickled_tuple = pickling.load(f, custom_main_path)
            if len(unpickled_tuple) == 4:
                fun, args, kwargs, meta_data = unpickled_tuple
                output_pickle_path = executor.format_outfile_name(cfut_dir, workerid_with_idx)
            else:
                assert len(unpickled_tuple) == 5, "Unexpected encoding"
                fun, args, kwargs, meta_data, output_pickle_path = unpickled_tuple

        if type(fun) == str:
            with open(fun, "rb") as function_file:
                fun = pickling.load(function_file, custom_main_path)

        setup_logging(meta_data)
        
        logging.info("Job computation started (jobid={}, workerid_with_idx={}).".format(executor.get_current_job_id(), workerid_with_idx))
        result = True, fun(*args, **kwargs)
        logging.info("Job computation completed.")
        out = pickling.dumps(result)

    except Exception as e:
        print(traceback.format_exc())

        result = False, format_remote_exc()
        logging.info("Job computation failed.")
        out = pickling.dumps(result)

    destfile = output_pickle_path
    tempfile = str(destfile) + ".tmp"
    with open(tempfile, "wb") as f:
        f.write(out)
    logging.info("Pickle file written to {}.".format(tempfile))
    os.rename(tempfile, destfile)
    logging.info("Pickle file renamed to {}.".format(destfile))


def setup_logging(meta_data):
    logging_config = meta_data.get("logging_config", {"level": logging.DEBUG, "format": "%(asctime)s %(levelname)s %(message)s"})

    # Call basicConfig which is necessary for the logging to work.
    logging.basicConfig(**logging_config)

    # It can happen that the pickled logger was already initialized. In this case,
    # the above basicConfig call was a noop. Therefore, we have to set the level explicitly.
    logger = logging.getLogger()
    if "level" in logging_config:
        logger.setLevel(logging_config["level"])

    logging.info("Setting up logging.basicConfig (potentially overwriting logging configuration of the main script). Config: {}".format(logging_config))
    logging.info("Starting job computation...")


if __name__ == "__main__":
    worker_id = sys.argv[1]
    cfut_dir = sys.argv[2]
    job_array_index = get_executor_class().get_job_array_index()

    worker(worker_id, job_array_index, cfut_dir)
    # This is a workaround for the case that some subprocesses are still hanging around and are waited for.
    # If this point is reached, results were written to disk and we can "safely" shut down everything.
    sys.exit()
