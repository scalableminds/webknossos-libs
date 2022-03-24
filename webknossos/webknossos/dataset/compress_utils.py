import logging
from pathlib import Path
from typing import Tuple

import wkw

from ..utils import time_start, time_stop, warn_deprecated


def compress_file_job(args: Tuple[Path, Path]) -> None:
    """Deprecated, please use `compress()` on a `MagView` instead."""
    warn_deprecated("compress_file_job", "MagView.compress()")
    source_path, target_path = args
    try:
        time_start("Compressing '{}' to '{}'".format(source_path, target_path))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        wkw.File.compress(str(source_path), str(target_path))

        if not target_path.exists():
            raise Exception("Did not create compressed file {}".format(target_path))

        time_stop("Compressing '{}' to '{}'".format(source_path, target_path))

    except Exception as exc:
        logging.error(
            "Compression of '{}' to '{}' failed with {}".format(
                source_path, target_path, exc
            )
        )
        raise exc
