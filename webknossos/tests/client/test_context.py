import pytest

from webknossos.client.context import _get_context, _WebknossosContext

pytestmark = [pytest.mark.with_vcr]


@pytest.fixture
def env_context() -> _WebknossosContext:
    return _get_context()


# pylint: disable=redefined-outer-name


def test_user_organization(env_context: _WebknossosContext) -> None:
    assert env_context.organization_id == "Organization_X"
