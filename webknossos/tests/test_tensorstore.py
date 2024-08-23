import difflib
from os import PathLike
from pathlib import Path
from time import strftime, gmtime
from typing import List, Optional

import networkx as nx
import pytest

import webknossos as wk
from pathlib import Path
from tempfile import TemporaryDirectory
from time import gmtime, strftime
from typing import Iterator

import numpy as np
import pytest

from .constants import TESTDATA_DIR
from .dataset.test_dataset_download_upload_remote import sample_dataset

@pytest.fixture(scope="module")
def sample_bbox() -> wk.BoundingBox:
    return wk.BoundingBox((2807, 4352, 1794), (10, 10, 10))

@pytest.fixture(scope="module")
def sample_dataset(sample_bbox: wk.BoundingBox) -> Iterator[wk.Dataset]:
    url = "https://webknossos.org/datasets/scalable_minds/l4_sample_dev"
    with TemporaryDirectory() as temp_dir:
        yield wk.Dataset.download(url, path=Path(temp_dir) / "ds", bbox=sample_bbox)


def test_remote_tensorstore(sample_dataset: wk.Dataset) -> None:
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    sample_dataset.upload(
        new_dataset_name=f"test_remote_metadata_{time_str}"
    )
