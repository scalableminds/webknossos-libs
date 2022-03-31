import json
import re
import warnings
from io import BytesIO
from os import makedirs
from shutil import rmtree
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import httpx
import pytest
from hypothesis import strategies as st
from vcr import VCR
from vcr.request import Request as VcrRequest
from vcr.stubs import httpx_stubs

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


### VCR.py / pytest-recording CONFIG


@pytest.fixture(scope="module")
def vcr_config() -> Dict[str, Any]:
    return {
        "filter_query_parameters": ["timestamp", "token"],
        "filter_headers": ["x-auth-token"],
    }


_REPLACE_IN_REQUEST_CONTENT = {
    r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}": "2000-01-01_00-00-00",
    r"\"uploadId\": \"[^\"]+\"": '"uploadId": "2000-01-01T00-00-00__0011"',
}

_REPLACE_IN_REQUEST_MULTIFORM = {
    r"\r\n": "\n",
    r"--[0-9a-f]+": "--",
    r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}__[0-9a-f\-]+": "2000-01-01T00-00-00__0011",
}


def _before_record_request(request: VcrRequest) -> VcrRequest:
    """This function formats and cleans request data to make it
    more readable and idempotent when re-snapshotting"""

    request.uri = re.sub(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "2000-01-01_00-00-00", request.uri
    )

    try:
        body_str = request.body.decode("utf-8")
    except Exception:
        body_str = None

    if "content-type" in request.headers:
        request.headers["content-type"] = re.sub(
            r"boundary=[0-9a-f]+",
            "boundary=fffffff0000000",
            request.headers["content-type"],
        )
        if (
            request.headers["content-type"].startswith("multipart/form-data")
            and body_str is not None
        ):
            for regex_to_replace, replace_with in _REPLACE_IN_REQUEST_MULTIFORM.items():
                body_str = re.sub(regex_to_replace, replace_with, body_str)

    if body_str is not None:
        for regex_to_replace, replace_with in _REPLACE_IN_REQUEST_CONTENT.items():
            body_str = re.sub(regex_to_replace, replace_with, body_str)
        request.body = body_str
    return request


_REPLACE_IN_RESPONSE_CONTENT = {
    r"\"lastUsedByUser\":\d+": '"lastUsedByUser":1010101010101',
    r"\"token\":\"[^\"]+\"": '"token":"xxxsecrettokenxxx"',
}


_HEADERS_TO_REMOVE = ["x-powered-by"]


def _before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """This function formats and cleans response data to make it
    more readable and idempotent when re-snapshotting"""

    # sort and lower-case all headers to stay consistent across servers, also cleanup:
    for key, value in sorted(response["headers"].items()):
        del response["headers"][key]
        if key.lower() not in _HEADERS_TO_REMOVE:
            if not (
                key.lower() == "connection" and (value == "close" or "close" in value)
            ):
                response["headers"][key.lower()] = value
    if "date" in response["headers"]:
        response["headers"]["date"] = "Mon, 01 Jan 2000 00:00:00 GMT"

    if isinstance(response["content"], str):
        for regex_to_replace, replace_with in _REPLACE_IN_RESPONSE_CONTENT.items():
            response["content"] = re.sub(
                regex_to_replace, replace_with, response["content"]
            )
        if "loggedTime" in response["content"]:
            json_content = json.loads(response["content"])
            json_content["loggedTime"] = [
                i
                for i in json_content["loggedTime"]
                if i["paymentInterval"]["year"] <= 2021
            ]
            response["content"] = json.dumps(json_content)

    return response


def pytest_recording_configure(config: Any, vcr: VCR) -> None:
    del config
    vcr.before_record_request = _before_record_request
    vcr.before_record_response = _before_record_response


# The remaining code of this module monkeypatches VCR with the following PRs:
# https://github.com/kevin1024/vcrpy/pull/574
# https://github.com/kevin1024/vcrpy/pull/583
#
# Additionally, we added some logic to unzip binary zip-blobs if possible


def _decode_if_possible(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            pass
    return value


def _clean_zip_file_content(filename: str, value: Any) -> Any:
    value = _handle_special_formats(value)
    if isinstance(value, str) and filename.endswith(".nml"):
        return re.sub(
            r"<meta name=\"timestamp\" content=\"\d+\" />",
            '<meta name="timestamp" content="1643210000000" />',
            value,
        )
    return value


def _unzip_if_possible(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            with BytesIO(value) as buffer:
                with ZipFile(buffer) as zipfile:
                    files = {}
                    for name in zipfile.namelist():
                        entry = zipfile.read(name)
                        files[name] = _clean_zip_file_content(name, entry)
                return {"zip": files}
        except Exception:
            pass
    return value


def _handle_special_formats(value: Any) -> Any:
    value = _unzip_if_possible(value)
    value = _decode_if_possible(value)
    return value


def _make_vcr_request(httpx_request: httpx.Request, **_kwargs: Any) -> VcrRequest:
    body = _handle_special_formats(httpx_request.read())
    if isinstance(body, bytes):
        partially_decoded = body.decode("utf-8", "ignore")
        if "Content-Type: application/octet-stream" in partially_decoded:
            partially_decoded = re.sub(
                r"Content-Type: application/octet-stream.*--\s*",
                "Content-Type: application/octet-stream<omitted> --",
                partially_decoded,
                flags=re.DOTALL,
            )
            body = partially_decoded

    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return VcrRequest(httpx_request.method, uri, body, headers)


httpx_stubs._make_vcr_request = _make_vcr_request


def _to_serialized_response(httpx_reponse: httpx.Response) -> Dict[str, Any]:
    return {
        "status_code": httpx_reponse.status_code,
        "http_version": httpx_reponse.http_version,
        "headers": httpx_stubs._transform_headers(httpx_reponse),
        "content": _handle_special_formats(httpx_reponse.content),
    }


httpx_stubs._to_serialized_response = _to_serialized_response


def _from_special_formats(value: Any) -> bytes:
    # zip-files were previously decoded as dicts with the key "zip",
    # see _unzip_if_possible()
    if isinstance(value, dict) and "zip" in value:
        with BytesIO() as buffer:
            with ZipFile(buffer, mode="a") as zipfile:
                for name, entry in value["zip"].items():
                    entry = _from_special_formats(entry)
                    zipfile.writestr(name, entry)
            return buffer.getvalue()

    if not isinstance(value, bytes):
        try:
            return value.encode("utf-8")
        except Exception:
            pass

    return value


@patch("httpx.Response.close", MagicMock())
@patch("httpx.Response.read", MagicMock())
def _from_serialized_response(
    request: httpx.Request, serialized_response: Any, history: Any = None
) -> httpx.Response:
    content = serialized_response.get("content")
    content = _from_special_formats(content)

    response = httpx.Response(
        status_code=serialized_response.get("status_code"),
        request=request,
        headers=httpx_stubs._from_serialized_headers(
            serialized_response.get("headers")
        ),
        content=content,
        history=history or [],
    )
    response._content = content
    return response


httpx_stubs._from_serialized_response = _from_serialized_response


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    # Automatically add the vcr marker to all tests
    # which don't already have it.
    for item in items:
        if item.get_closest_marker("vcr") is None:
            item.add_marker("vcr")

        # To allow for UNIX socket communication necessary for spawn multiprocessing
        # addresses starting with `/` are allowed
        marker = item.get_closest_marker("block_network")
        if marker is None:
            new_marker = pytest.mark.block_network(allowed_hosts=["/.*"])
            item.add_marker(new_marker)
        else:
            marker.kwargs["allowed_hosts"].append("/.*")
