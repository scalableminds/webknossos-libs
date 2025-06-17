import gc
import os
import warnings
from collections.abc import Generator
from os import makedirs
from pathlib import Path
from shutil import rmtree, unpack_archive
from typing import Any

import fsspec.implementations.http as http
import httpx
import pytest
from hypothesis import strategies as st

import webknossos as wk
from webknossos.client._upload_dataset import _cached_get_upload_datastore
from webknossos.client.context import _clear_all_context_caches
from webknossos.webknossos.dataset._array import _clear_tensorstore_context

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


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


# fsspec uses aiohttp for http paths, but aiohttp does not consider environment variables by default
# as we need to set the HTTP_PROXY environment variable for proxay, we need to monkeypatch the fsspec http implementation
@pytest.fixture(autouse=True)
def aiohttp_use_env_variables(monkeypatch: pytest.MonkeyPatch) -> Generator:
    import aiohttp

    class PatchedClientSession(aiohttp.ClientSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "trust_env" not in kwargs:
                kwargs["trust_env"] = True
            super().__init__(*args, **kwargs)

    # When aiohttp's ClientSession is imported in tests, it will be replaced by PatchedClientSession
    # PatchedClientSession will behave like the original ClientSession, but with trust_env set to True by default
    monkeypatch.setattr(http.aiohttp, "ClientSession", PatchedClientSession)
    yield


@pytest.fixture(autouse=True, scope="function")
def clear_testoutput() -> Generator:
    makedirs(TESTOUTPUT_DIR, exist_ok=True)
    yield
    rmtree(TESTOUTPUT_DIR)


@pytest.fixture(autouse=True, scope="function")
def clear_caches() -> Generator:
    _clear_all_context_caches()
    _cached_get_upload_datastore.cache_clear()
    _clear_tensorstore_context()
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


@pytest.fixture(autouse=True, scope="function")
def use_replay_proxay(request: Any) -> Generator:
    testname = f"{request.node.parent.name.removesuffix('.py')}/{request.node.name.replace('/', '__')}"
    if "use_proxay" in request.keywords:
        os.environ["HTTP_PROXY"] = "http://localhost:3000"
        os.environ["http_proxy"] = (
            "http://localhost:3000"  # for tensorstore. env var names are case-sensitive on Linux
        )
        httpx.post("http://localhost:3000/__proxay/tape", json={"tape": testname})
    yield
    if "HTTP_PROXY" in os.environ:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("http_proxy", None)


### Misc fixtures


@pytest.fixture(scope="session")
def WT1_path() -> Path:
    ds_path = TESTDATA_DIR / "WT1_wkw"
    if ds_path.exists():
        rmtree(ds_path)
    unpack_archive(
        TESTDATA_DIR / "WT1_wkw.tar.gz",
        ds_path,
    )
    return ds_path
