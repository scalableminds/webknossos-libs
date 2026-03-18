import gc
import os
import sys
import warnings
from collections.abc import Generator, Iterator
from pathlib import Path
from shutil import unpack_archive
from typing import Any

import fsspec.implementations.http as http
import httpx
import pytest
from hypothesis import strategies as st
from upath import UPath

import webknossos as wk
from webknossos.client._upload_dataset import _cached_get_upload_datastore
from webknossos.client.context import _clear_all_context_caches
from webknossos.utils import rmtree

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("skip_on_windows") and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="not supported on Windows"))


@pytest.fixture()
def tmp_upath(tmp_path: Path) -> Iterator[UPath]:
    yield UPath(tmp_path)


def pytest_make_parametrize_id(config: Any, val: Any, argname: str) -> Any:
    del config
    del argname
    if isinstance(val, str):
        val = val.rsplit("?", maxsplit=1)[0]
        val = val.rsplit("#", maxsplit=1)[0]
        parts = val.rstrip("/").split("/")
        take = 1
        while (len(parts[-take]) <= 1 or parts[-take] == "view") and take < len(parts):
            take += 1
        return "/".join(parts[-take:])
    # return None to let pytest handle the formatting
    return None


@pytest.fixture(autouse=True)
def ensure_gc() -> None:
    gc.collect()


### HYPOTHESIS STRATEGIES (library to test many combinations for data class input)


_vec3_int_strategy = st.builds(wk.Vec3Int, st.integers(), st.integers(), st.integers())

st.register_type_strategy(wk.Vec3Int, _vec3_int_strategy)

_positive_vec3_int_strategy = st.builds(
    wk.Vec3Int,
    st.integers(min_value=0),
    st.integers(min_value=0),
    st.integers(min_value=0),
)

st.register_type_strategy(
    wk.BoundingBox,
    st.builds(wk.BoundingBox, _positive_vec3_int_strategy, _positive_vec3_int_strategy),
)

_mag_strategy = st.builds(
    lambda mag_xy_log2, mag_z_log2: wk.Mag(
        (2**mag_xy_log2, 2**mag_xy_log2, 2**mag_z_log2)
    ),
    st.integers(min_value=0, max_value=12),
    st.integers(min_value=0, max_value=12),
)

st.register_type_strategy(wk.Mag, _mag_strategy)


### PYTEST SETUP & TEARDOWN


@pytest.fixture(autouse=True, scope="function")
def clear_testoutput() -> Generator:
    TESTOUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    rmtree(TESTOUTPUT_DIR)


@pytest.fixture(autouse=True, scope="function")
def clear_caches() -> Generator:
    _clear_all_context_caches()
    _cached_get_upload_datastore.cache_clear()
    yield


@pytest.fixture(autouse=True, scope="function")
def error_on_deprecations() -> Generator:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", module="webknossos", message=r"\[DEPRECATION\]"
        )
        yield


@pytest.fixture(autouse=True, scope="function")
def error_on_warnings() -> Generator:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", module="webknossos", message=r"\[WARNING\]")
        yield


### Misc fixtures


@pytest.fixture(scope="session")
def WT1_upath() -> UPath:
    ds_path = TESTDATA_DIR / "WT1_wkw"
    if ds_path.exists():
        rmtree(ds_path)
    unpack_archive(str(TESTDATA_DIR / "WT1_wkw.tar.gz"), str(ds_path))
    return ds_path
