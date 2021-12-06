from os import makedirs
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, Generator, Union
from unittest.mock import MagicMock, patch

import httpx
import pytest
from vcr.request import Request as VcrRequest
from vcr.stubs import httpx_stubs

from webknossos.client.context import _clear_all_context_caches

TESTOUTPUT_DIR = Path("testoutput")


@pytest.fixture(autouse=True, scope="function")
def run_around_tests() -> Generator:
    makedirs(TESTOUTPUT_DIR, exist_ok=True)
    _clear_all_context_caches()
    yield
    rmtree(TESTOUTPUT_DIR)


@pytest.fixture(scope="module")
def vcr_config() -> Dict[str, Any]:
    return {
        "filter_query_parameters": ["timestamp", "token"],
        "filter_headers": ["x-auth-token"],
    }


# The remaining code of this module monkeypatches VCR with the following PRs:
# https://github.com/kevin1024/vcrpy/pull/574
# https://github.com/kevin1024/vcrpy/pull/583


def _make_vcr_request(httpx_request: httpx.Request, **_kwargs: Any) -> VcrRequest:
    body = httpx_request.read()
    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return VcrRequest(httpx_request.method, uri, body, headers)


httpx_stubs._make_vcr_request = _make_vcr_request


def _to_serialized_response(httpx_reponse: httpx.Response) -> Dict[str, Any]:
    content: Union[bytes, str] = httpx_reponse.content
    try:
        if content is not None and not isinstance(content, str):
            content = content.decode("utf-8")
    except (TypeError, UnicodeDecodeError, AttributeError):
        # Sometimes the string actually is binary or StringIO object,
        # so if you can't decode it, just give up.
        pass

    return {
        "status_code": httpx_reponse.status_code,
        "http_version": httpx_reponse.http_version,
        "headers": httpx_stubs._transform_headers(httpx_reponse),
        "content": content,
    }


httpx_stubs._to_serialized_response = _to_serialized_response


@patch("httpx.Response.close", MagicMock())
@patch("httpx.Response.read", MagicMock())
def _from_serialized_response(
    request: httpx.Request, serialized_response: Any, history: Any = None
) -> httpx.Response:
    content = serialized_response.get("content")
    try:
        if content is not None and not isinstance(content, bytes):
            content = content.encode("utf-8")
    except (TypeError, UnicodeEncodeError, AttributeError):
        # sometimes the thing actually is binary, so if you can't encode
        # it, just give up.
        pass

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
