import logging
from os import makedirs, path
from pathlib import Path
from typing import Tuple

import wkw

from ..utils import time_start, time_stop


def compress_file_job(args: Tuple[Path, Path]) -> None:
    source_path, target_path = args
    try:
        time_start("Compressing '{}' to '{}'".format(source_path, target_path))

        makedirs(path.dirname(target_path), exist_ok=True)
        wkw.File.compress(str(source_path), str(target_path))

        if not path.exists(target_path):
            raise Exception("Did not create compressed file {}".format(target_path))

        time_stop("Compressing '{}' to '{}'".format(source_path, target_path))

    except Exception as exc:
        logging.error(
            "Compression of '{}' to '{}' failed with {}".format(
                source_path, target_path, exc
            )
        )
        raise exc
