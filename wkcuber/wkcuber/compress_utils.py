import time
import logging
from os import path, makedirs
from pathlib import Path
from typing import Tuple

import wkw


def compress_file_job(args: Tuple[Path, Path]) -> None:
    source_path, target_path = args
    try:
        logging.debug("Compressing '{}' to '{}'".format(source_path, target_path))
        ref_time = time.time()

        makedirs(path.dirname(target_path), exist_ok=True)
        wkw.File.compress(str(source_path), str(target_path))

        if not path.exists(target_path):
            raise Exception("Did not create compressed file {}".format(target_path))

        logging.debug(
            "Compressing of '{}' took {:.8f}s".format(
                source_path, time.time() - ref_time
            )
        )
    except Exception as exc:
        logging.error("Compressing of '{}' failed with {}".format(source_path, exc))
        raise exc
