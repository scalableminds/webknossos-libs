import warnings
from os import makedirs
from shutil import rmtree
from typing import Generator

import pytest
from hypothesis import strategies as st

import webknossos as wk
from webknossos.client.context import _clear_all_context_caches

from .constants import TESTOUTPUT_DIR

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
        (2 ** mag_xy_log2, 2 ** mag_xy_log2, 2 ** mag_z_log2)
    ),
    st.integers(min_value=0, max_value=12),
    st.integers(min_value=0, max_value=12),
)

st.register_type_strategy(wk.Mag, _mag_strategy)


### PYTEST SETUP & TEARDOWN


@pytest.fixture(autouse=True, scope="function")
def clear_testoutput() -> Generator:
    makedirs(TESTOUTPUT_DIR, exist_ok=True)
    yield
    rmtree(TESTOUTPUT_DIR)


@pytest.fixture(autouse=True, scope="function")
def clear_context_caches() -> Generator:
    _clear_all_context_caches()
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
