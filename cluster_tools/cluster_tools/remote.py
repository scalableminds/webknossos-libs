"""Tools for executing remote commands."""
import sys
import os
import traceback
from .util import local_filename
from . import pickling
import logging

INFILE_FMT = local_filename("cfut.in.%s.pickle")
OUTFILE_FMT = local_filename("cfut.out.%s.pickle")


def format_remote_exc():
    typ, value, tb = sys.exc_info()
    tb = tb.tb_next  # Remove root call to worker().
    return "".join(traceback.format_exception(typ, value, tb))


def worker(workerid):
    """Called to execute a job on a remote host."""
    try:
        with open(INFILE_FMT % workerid, "rb") as f:
            indata = f.read()
        fun, args, kwargs, meta_data = pickling.loads(indata)
  
        setup_logging(meta_data)
        result = True, fun(*args, **kwargs)
        logging.info("Job computation completed.")
        out = pickling.dumps(result, True)

    except Exception as e:
        print(traceback.format_exc())

        result = False, format_remote_exc()
        logging.info("Job computation failed.")
        out = pickling.dumps(result, False)


    destfile = OUTFILE_FMT % workerid
    tempfile = destfile + ".tmp"
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
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if task_id is not None:
        worker_id = worker_id + "_" + task_id

    worker(worker_id)
