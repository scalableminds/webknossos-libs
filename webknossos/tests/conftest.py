import gc
import sys
import warnings
from collections.abc import Generator, Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest
from cluster_tools import Executor, get_executor
from hypothesis import strategies as st
from upath import UPath

import webknossos as wk
from webknossos.client._upload_dataset import _cached_get_upload_datastore
from webknossos.client.context import _clear_all_context_caches
from webknossos.utils import rmtree

from .constants import TESTOUTPUT_DIR, use_moto


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


@pytest.fixture()
def ensure_gc() -> None:
    """Opt-in full collection before a test.

    Only worth its cost (~40-100ms per test) in modules that allocate large
    image buffers. Request it per module with
    `pytestmark = pytest.mark.usefixtures("ensure_gc")`.
    """
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


@pytest.fixture(scope="session")
def shared_executor() -> Iterator[Executor]:
    """One process pool for the whole session.

    Library functions that take an optional `executor` (`Layer.downsample`,
    `MagView.rechunk`, `View.map_chunk`, ...) build a fresh one whenever the
    caller passes none. Constructing it is free, but the first submit spawns
    the workers, which costs ~1s for two of them and ~1.4s for `cpu_count()`
    on a 10-core machine. Reusing one pool pays that once per session instead
    of once per call; a warm submit is well under a millisecond.
    """
    with get_executor("multiprocessing", max_workers=2) as executor:
        yield executor


@pytest.fixture(autouse=True)
def reuse_executor(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hands the shared pool to every `utils.wrap_executor` call.

    Opt out with `@pytest.mark.own_executor` where the executor lifecycle
    itself is under test.
    """
    if request.node.get_closest_marker("own_executor"):
        return

    def wrap_executor(executor: Executor | None = None) -> AbstractContextManager:
        # Resolved lazily so tests that never reach one of those functions
        # don't pay for starting the worker processes.
        return nullcontext(
            executor
            if executor is not None
            else request.getfixturevalue("shared_executor")
        )

    monkeypatch.setattr(wk.utils, "wrap_executor", wrap_executor)


@pytest.fixture(scope="session")
def moto_server() -> Generator:
    """One in-process S3 server per test session.

    Opt in per module with `pytestmark = pytest.mark.usefixtures("moto_server")`.
    """
    with use_moto():
        yield


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
