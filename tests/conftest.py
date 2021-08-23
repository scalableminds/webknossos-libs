import pytest
from pathlib import Path
from os import makedirs
from shutil import rmtree
from typing import Generator

TESTOUTPUT_DIR = Path("testoutput")


@pytest.fixture(autouse=True, scope="function")
def run_around_tests() -> Generator:
    makedirs(TESTOUTPUT_DIR, exist_ok=True)
    yield
    rmtree(TESTOUTPUT_DIR)
    print("CLEANUP")
