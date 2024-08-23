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
from .dataset.test_dataset_download_upload_remote import sample_dataset, sample_bbox


def test_remote_tensorstore(sample_dataset: wk.Dataset):
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    sample_dataset.upload(
        new_dataset_name=f"test_remote_metadata_{time_str}"
    )
