import pytest

from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client._generated.api.default import (
    annotation_info,
    build_info,
    current_user_info,
    dataset_info,
    datastore_list,
    generate_token_for_data_store,
    health,
    user_list,
    user_logged_time,
)
from webknossos.client._generated.client import Client
from webknossos.client.apiclient import WkApiClient
from webknossos.client.apiclient.models import ApiDataStore
from webknossos.client._generated.models.datastore_list_response_200_item import (
    DatastoreListResponse200Item,
)
from webknossos.client._generated.types import Unset
from webknossos.client.context import _get_api_client, webknossos_context
from webknossos.utils import time_since_epoch_in_ms

pytestmark = [pytest.mark.with_vcr]

DATASTORE_URL = "https://data-humerus.webknossos.org"


@pytest.fixture
def client() -> WkApiClient:
    return _get_api_client()


@pytest.fixture
def auth_client() -> WkApiClient:
    return _get_api_client(enforce_auth=True)


# pylint: disable=redefined-outer-name


def test_health(client: WkApiClient) -> None:
    # No exception should be raised
    client.health()


def test_annotation_info(auth_client: WkApiClient) -> None:
    annotation_id = "570ba0092a7c0e980056fe9b"  # pylint: disable=redefined-builtin
    typ = "Explorational"
    api_annotation = auth_client.annotation_info(annotation_id)
    assert api_annotation.id == id
    assert api_annotation.typ == typ


def test_datastore_list(auth_client: WkApiClient) -> None:
    datastores = auth_client.datastore_list()
    internal_datastore = ApiDataStore(
        name="localhost",
        url="http://localhost:9000",
        allows_upload=True,
    )
    assert internal_datastore in datastores


def test_generate_token_for_data_store(auth_client: Client) -> None:
    generate_token_for_data_store_response = generate_token_for_data_store.sync(
        client=auth_client
    )
    assert generate_token_for_data_store_response is not None
    assert len(generate_token_for_data_store_response.token) > 0


def test_current_user_info_and_user_logged_time(auth_client: Client) -> None:
    current_user_info_response = current_user_info.sync(client=auth_client)
    assert current_user_info_response is not None
    assert len(current_user_info_response.email) > 0
    assert not isinstance(current_user_info_response.teams, Unset)
    assert len(current_user_info_response.teams) > 0
    assert current_user_info_response.is_active
    user_logged_time_response = user_logged_time.sync(
        id=current_user_info_response.id, client=auth_client
    )
    assert user_logged_time_response is not None
    assert isinstance(user_logged_time_response.logged_time, list)


def test_user_list(auth_client: Client) -> None:
    user_list_response = user_list.sync(client=auth_client)
    assert isinstance(user_list_response, list)


def test_dataset_info() -> None:
    with webknossos_context(url=DEFAULT_WEBKNOSSOS_URL):
        client = _get_api_client()
    api_dataset = client.dataset_info(
        organization_name="scalable_minds",
        dataset_name="l4dense_motta_et_al_demo"
    )
    assert api_dataset.data_store.url == DATASTORE_URL
    assert api_dataset.display_name == "L4 Mouse Cortex Demo"
    data_layers = api_dataset.data_source.data_layers
    assert not isinstance(data_layers, Unset)
    assert sorted(
        (layer.name, layer.category, layer.element_class) for layer in data_layers
    ) == [
        ("color", "color", "uint8"),
        ("predictions", "color", "uint24"),
        ("segmentation", "segmentation", "uint32"),
    ]


def test_build_info(client: Client) -> None:
    response = build_info.sync(
        client=client,
    )
    assert response is not None
    wk_info = response.webknossos
    wk_wrap_info = response.webknossos_wrap
    assert not isinstance(wk_info, Unset)
    assert not isinstance(wk_wrap_info, Unset)
    assert wk_info.name == "webknossos"
    assert wk_wrap_info.name == "webknossos-wrap"
    assert response.local_data_store_enabled
    assert response.local_tracing_store_enabled
