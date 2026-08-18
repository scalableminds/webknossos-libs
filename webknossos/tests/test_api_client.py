import pytest

from webknossos.client.api_client import WkApiClient
from webknossos.client.api_client.datastore_api_client import DatastoreApiClient
from webknossos.client.api_client.models import ApiDataStore
from webknossos.client.context import _get_api_client
from webknossos.dataset_properties import DatasetProperties
from webknossos.geometry import Mag

pytestmark = [pytest.mark.skip_on_windows]

DATASTORE_URL = "http://localhost:9000"


@pytest.fixture
def client() -> WkApiClient:
    return _get_api_client()


def test_health(client: WkApiClient) -> None:
    # No exception should be raised
    client.health()


def test_annotation_info(client: WkApiClient) -> None:
    annotation_id = "570ba0092a7c0e980056fe9b"
    typ = "Explorational"
    api_annotation = client.annotation_info(annotation_id=annotation_id)
    assert api_annotation.id == annotation_id
    assert api_annotation.typ == typ


def test_datastore_list(client: WkApiClient) -> None:
    datastores = client.datastore_list()
    internal_datastore = ApiDataStore(
        name="localhost",
        url="http://localhost:9000",
        allows_upload=True,
    )
    assert internal_datastore in datastores


def test_current_user_info_and_user_logged_time(client: WkApiClient) -> None:
    current_api_user = client.user_current()

    assert len(current_api_user.email) > 0
    assert len(current_api_user.teams) > 0
    assert current_api_user.is_active
    user_logged_time_response = client.user_logged_time(user_id=current_api_user.id)
    assert user_logged_time_response is not None
    assert isinstance(user_logged_time_response.logged_time, list)


def test_user_list(client: WkApiClient) -> None:
    api_users = client.user_list()
    assert isinstance(api_users, list)


def test_dataset_info(client: WkApiClient) -> None:
    dataset_id = client.dataset_id_from_name(
        directory_name="l4_sample", organization_id="Organization_X"
    )
    api_dataset = client.dataset_info(
        dataset_id=dataset_id,
    )
    assert api_dataset.data_store.url == DATASTORE_URL
    data_source = api_dataset.data_source
    assert isinstance(data_source, DatasetProperties)
    data_layers = data_source.data_layers
    assert data_layers is not None
    assert sorted(
        (layer.name, layer.category, layer.dtype) for layer in data_layers
    ) == [
        ("color", "color", "uint8"),
        ("segmentation", "segmentation", "uint32"),
    ]


def test_build_info(client: WkApiClient) -> None:
    api_build_info = client.build_info()
    assert api_build_info.webknossos.name == "webknossos"
    assert api_build_info.local_data_store_enabled
    assert api_build_info.local_tracing_store_enabled


def test_datastore_api_client_paths() -> None:
    """These routes are also used to build RemoteMagView/RemoteAttachments paths."""
    client = DatastoreApiClient(datastore_base_url=DATASTORE_URL, timeout_seconds=30)
    assert client.url_prefix == f"{DATASTORE_URL}/data/v{client.webknossos_api_version}"

    dataset_id = "59e9cfbdba632ac2ab8b23b5"
    annotation_id = "570ba0092a7c0e980056fe9b"
    assert (
        client.zarr_streaming_dataset_url(dataset_id)
        == f"{client.url_prefix}/zarr/{dataset_id}/"
    )
    assert (
        client.zarr_streaming_annotation_url(annotation_id)
        == f"{client.url_prefix}/annotations/zarr/{annotation_id}/"
    )
    assert (
        client.proxy_dataset_url(dataset_id)
        == f"{client.url_prefix}/datasets/{dataset_id}/proxy/"
    )

    mag = Mag(1)
    assert client.zarr_streaming_mag_path("color", mag) == "color/1"
    assert client.proxy_mag_path("color", mag) == "layers/color/mags/1"
    assert (
        client.proxy_attachment_path("segmentation", "agglomerate", "map_all")
        == "layers/segmentation/attachments/agglomerate/map_all"
    )
