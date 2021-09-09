import calendar
from datetime import datetime

from webknossos.client import _get_generated_client
from webknossos.client.generated.api.default import info, list_
from webknossos.client.generated.models.list_response_200_item import (
    ListResponse200Item,
)
from webknossos.client.generated.types import Unset


def time_since_epoch_in_ms() -> int:
    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    return unixtime * 1000


WK_URL = "https://webknossos.org"

client = _get_generated_client()
auth_client = _get_generated_client(enforce_token=True)


def test_annotation_info() -> None:
    id = "6114d9410100009f0096c640"
    typ = "Explorational"
    info_object = info.sync(
        typ=typ, id=id, client=client, timestamp=time_since_epoch_in_ms()
    )
    assert info_object is not None
    assert info_object.id == id
    assert info_object.typ == typ
    assert not isinstance(info_object.data_store, Unset)
    assert info_object.data_store.url == client.base_url


def test_list_datastores() -> None:
    datastores = list_.sync(client=auth_client)
    internal_datastore = ListResponse200Item(
        name="webknossos.org",
        url=WK_URL,
        is_foreign=False,
        is_scratch=False,
        is_connector=False,
        allows_upload=True,
    )
    assert datastores is not None
    assert internal_datastore in datastores
