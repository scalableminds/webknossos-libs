import pytest

from webknossos.client.context import get_generated_client
from webknossos.client.defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client.generated.api.default import (
    annotation_info,
    build_info,
    dataset_info,
    datastore_list,
    health,
)
from webknossos.client.generated.client import Client
from webknossos.client.generated.models.datastore_list_response_200_item import (
    DatastoreListResponse200Item,
)
from webknossos.client.generated.types import Unset
from webknossos.utils import time_since_epoch_in_ms


@pytest.fixture
def client() -> Client:
    return get_generated_client()


@pytest.fixture
def auth_client() -> Client:
    return get_generated_client(enforce_auth=True)


@pytest.mark.vcr()
def test_health(client: Client) -> None:
    response = health.sync_detailed(client=client)
    assert response.status_code == 200


@pytest.mark.vcr()
def test_annotation_info(client: Client) -> None:
    id = "6114d9410100009f0096c640"
    typ = "Explorational"
    info_object = annotation_info.sync(
        typ=typ, id=id, client=client, timestamp=time_since_epoch_in_ms()
    )
    assert info_object is not None
    assert info_object.id == id
    assert info_object.typ == typ
    assert not isinstance(info_object.data_store, Unset)
    assert info_object.data_store.url == client.base_url


@pytest.mark.vcr()
def test_datastore_list(auth_client: Client) -> None:
    datastores = datastore_list.sync(client=auth_client)
    internal_datastore = DatastoreListResponse200Item(
        name="webknossos.org",
        url=DEFAULT_WEBKNOSSOS_URL,
        is_foreign=False,
        is_scratch=False,
        is_connector=False,
        allows_upload=True,
    )
    assert datastores is not None
    assert internal_datastore in datastores


@pytest.mark.vcr()
def test_dataset_info(client: Client) -> None:
    response = dataset_info.sync(
        organization_name="scalable_minds",
        data_set_name="l4dense_motta_et_al_demo",
        client=client,
    )
    assert response is not None
    assert not isinstance(response.data_store, Unset)
    assert response.data_store.url == client.base_url
    assert response.display_name == "L4 Mouse Cortex Demo"
    assert not isinstance(response.data_source, Unset)
    assert not isinstance(response.data_source.data_layers, Unset)
    assert sorted(
        (layer.name, layer.category, layer.element_class)
        for layer in response.data_source.data_layers
    ) == [
        ("color", "color", "uint8"),
        ("predictions", "color", "uint24"),
        ("segmentation", "segmentation", "uint32"),
    ]


@pytest.mark.vcr()
def test_build_info(client: Client) -> None:
    response = build_info.sync(
        client=client,
    )
    assert response is not None
    assert not isinstance(response.webknossos, Unset)
    assert not isinstance(response.webknossos_wrap, Unset)
    assert response.webknossos.name == "webknossos"
    assert response.webknossos_wrap.name == "webknossos-wrap"
    assert response.local_data_store_enabled
    assert response.local_tracing_store_enabled
