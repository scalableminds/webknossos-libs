from pathlib import Path
from shutil import rmtree
from typing import Union

import numpy as np

from webknossos.dataset import Dataset
from webknossos.utils import get_chunks
from wkcuber.utils import get_regular_chunks
import wkw
from wkcuber.mag import Mag
import os

BLOCK_LEN = 32

TESTOUTPUT_DIR = Path("testoutput")


def delete_dir(relative_path: Union[str, Path]) -> None:
    if os.path.exists(relative_path) and os.path.isdir(relative_path):
        rmtree(relative_path)


def test_get_chunks() -> None:
    source = list(range(0, 48))
    target = list(get_chunks(source, 8))

    assert len(target) == 6
    assert target[0] == list(range(0, 8))


def test_get_regular_chunks() -> None:
    target = list(get_regular_chunks(4, 44, 8))

    assert len(target) == 6
    assert list(target[0]) == list(range(0, 8))
    assert list(target[-1]) == list(range(40, 48))


def test_get_regular_chunks_max_inclusive() -> None:
    target = list(get_regular_chunks(4, 44, 1))

    assert len(target) == 41
    assert list(target[0]) == list(range(4, 5))
    # The last chunk should include 44
    assert list(target[-1]) == list(range(44, 45))
