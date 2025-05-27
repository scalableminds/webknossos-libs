import pytest

from webknossos import Tracingstore


@pytest.mark.use_proxay
def test_get_tracingstore() -> None:
    tracingstore = Tracingstore.get_tracingstore()

    assert tracingstore is not None
    assert tracingstore.url == "http://localhost:9000"
    assert tracingstore.name == "localhost"
