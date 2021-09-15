from os import makedirs
from pathlib import Path
from shutil import rmtree
from typing import Generator

import pytest

TESTOUTPUT_DIR = Path("testoutput")


@pytest.fixture(autouse=True, scope="function")
def run_around_tests() -> Generator:
    makedirs(TESTOUTPUT_DIR, exist_ok=True)
    yield
    rmtree(TESTOUTPUT_DIR)
