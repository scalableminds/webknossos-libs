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
from webknossos.client._generated.models.datastore_list_response_200_item import (
    DatastoreListResponse200Item,
)
from webknossos.client.context import _get_generated_client, webknossos_context
from webknossos.utils import time_since_epoch_in_ms

DATASTORE_URL = "https://data-humerus.webknossos.org"


@pytest.fixture
def client() -> Client:
    return _get_generated_client()


@pytest.fixture
def auth_client() -> Client:
    return _get_generated_client(enforce_auth=True)


# pylint: disable=redefined-outer-name


def test_health(client: Client) -> None:
    response = health.sync_detailed(client=client)
    assert response.status_code == 200


def test_annotation_info(auth_client: Client) -> None:
    id = "570ba0092a7c0e980056fe9b"  # pylint: disable=redefined-builtin
    typ = "Explorational"
    info_object = annotation_info.sync(
        typ=typ, id=id, client=auth_client, timestamp=time_since_epoch_in_ms()
    )
    assert info_object is not None
    assert info_object.id == id
    assert info_object.typ == typ
    assert info_object.data_store.url == auth_client.base_url


def test_datastore_list(auth_client: Client) -> None:
    datastores = datastore_list.sync(client=auth_client)
    internal_datastore = DatastoreListResponse200Item(
        name="localhost",
        url="http://localhost:9000",
        is_foreign=False,
        is_scratch=False,
        is_connector=False,
        allows_upload=True,
    )
    assert datastores is not None
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
        client = _get_generated_client()
    response = dataset_info.sync(
        organization_name="scalable_minds",
        data_set_name="l4dense_motta_et_al_demo",
        client=client,
    )
    assert response is not None
    assert response.data_store.url == DATASTORE_URL
    assert response.display_name == "L4 Mouse Cortex Demo"
    assert sorted(
        (layer.name, layer.category, layer.element_class)
        for layer in response.data_source.data_layers
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
    assert response.webknossos.name == "webknossos"
    assert response.webknossos_wrap.name == "webknossos-wrap"
    assert response.local_data_store_enabled
    assert response.local_tracing_store_enabled
