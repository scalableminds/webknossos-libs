import pytest

from webknossos.client.api_client import WkApiClient
from webknossos.client.api_client.models import ApiDataStore
from webknossos.client.context import _get_api_client

pytestmark = [pytest.mark.use_proxay]

DATASTORE_URL = "http://localhost:9000"


@pytest.fixture
def client() -> WkApiClient:
    return _get_api_client()


@pytest.fixture
def auth_client() -> WkApiClient:
    return _get_api_client(enforce_auth=True)


def test_health(client: WkApiClient) -> None:
    # No exception should be raised
    client.health()


def test_annotation_info(auth_client: WkApiClient) -> None:
    annotation_id = "570ba0092a7c0e980056fe9b"
    typ = "Explorational"
    api_annotation = auth_client.annotation_info(annotation_id)
    assert api_annotation.id == annotation_id
    assert api_annotation.typ == typ


def test_datastore_list(auth_client: WkApiClient) -> None:
    datastores = auth_client.datastore_list()
    internal_datastore = ApiDataStore(
        name="localhost",
        url="http://localhost:9000",
        allows_upload=True,
    )
    assert internal_datastore in datastores


def test_generate_token_for_data_store(auth_client: WkApiClient) -> None:
    api_datastore_token = auth_client.token_generate_for_data_store()
    assert len(api_datastore_token.token) > 0


def test_current_user_info_and_user_logged_time(auth_client: WkApiClient) -> None:
    current_api_user = auth_client.user_current()

    assert len(current_api_user.email) > 0
    assert len(current_api_user.teams) > 0
    assert current_api_user.is_active
    user_logged_time_response = auth_client.user_logged_time(current_api_user.id)
    assert user_logged_time_response is not None
    assert isinstance(user_logged_time_response.logged_time, list)


def test_user_list(auth_client: WkApiClient) -> None:
    api_users = auth_client.user_list()
    assert isinstance(api_users, list)


def test_dataset_info() -> None:
    client = _get_api_client()
    dataset_id = client.dataset_id_from_name("l4_sample", "Organization_X")
    api_dataset = client.dataset_info(
        dataset_id=dataset_id,
    )
    assert api_dataset.data_store.url == DATASTORE_URL
    data_layers = api_dataset.data_source.data_layers
    assert data_layers is not None
    assert sorted(
        (layer.name, layer.category, layer.element_class) for layer in data_layers
    ) == [
        ("color", "color", "uint8"),
        ("segmentation", "segmentation", "uint32"),
    ]


def test_build_info(client: WkApiClient) -> None:
    api_build_info = client.build_info()
    assert api_build_info.webknossos.name == "webknossos"
    assert api_build_info.local_data_store_enabled
    assert api_build_info.local_tracing_store_enabled
