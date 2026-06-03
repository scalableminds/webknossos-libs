import pytest

from webknossos.client.api_client import WkApiClient, WkApiClientV13
from webknossos.client.context import (
    _get_context,
    _WebknossosContext,
    login,
    webknossos_context,
)

pytestmark = [pytest.mark.skip_on_windows]


@pytest.fixture
def env_context() -> _WebknossosContext:
    return _get_context()


def test_user_organization(env_context: _WebknossosContext) -> None:
    assert env_context.organization_id == "Organization_X"


def test_trailing_slash_in_url(env_context: _WebknossosContext) -> None:
    with webknossos_context(url=env_context.url + "/"):
        assert env_context.url == _get_context().url


def test_login() -> None:
    # Use webknossos_context to isolate changes to the global context
    with webknossos_context():
        login(url="https://example.com", token="test_token")
        assert _get_context().url == "https://example.com"
        assert _get_context().token == "test_token"

        # Nested webknossos_context overrides login, then restores it on exit
        with webknossos_context(token="nested_token"):
            assert _get_context().token == "nested_token"
        assert _get_context().token == "test_token"


@pytest.mark.parametrize(
    "api_version,api_client_class", [(13, WkApiClientV13), (14, WkApiClient)]
)
def test_webknossos_context_api_version(
    api_version: int, api_client_class: type
) -> None:
    with webknossos_context(api_version=api_version):
        assert _get_context().api_version == api_version
        assert isinstance(_get_context().api_client, api_client_class)
