from pathlib import Path
from unittest.mock import patch

import pytest

import webknossos as wk
from webknossos.client import _upload_dataset
from webknossos.client.context import _WebknossosContext

pytestmark = [pytest.mark.skip_on_windows]


def test_make_resumable_without_token() -> None:
    resumable = _upload_dataset._make_resumable(
        target="http://localhost:9000/data/datasets",
        simultaneous_uploads=1,
        query={},
        context=_WebknossosContext(url="http://localhost:9000", token=None),
        upload_id="upload_id",
    )
    assert "X-Auth-Token" not in resumable.client.headers


def test_upload_with_linked_layers_passes_all_parameters(tmp_path: Path) -> None:
    ds = wk.Dataset(tmp_path / "ds", voxel_size=(1, 1, 1))
    ds.add_layer("color", wk.COLOR_CATEGORY)
    ds.add_layer("linked", wk.COLOR_CATEGORY)

    upload_dataset = _upload_dataset.upload_dataset
    with (
        patch.object(
            _upload_dataset, "upload_dataset", return_value="new_id"
        ) as recursive_call,
        pytest.warns(UserWarning, match="Excluding the following layers"),
    ):
        upload_dataset(
            ds,
            new_dataset_name="new_name",
            layers_to_link=[wk.LayerToLink(dataset_id="other", layer_name="linked")],
            jobs=2,
            datastore_url="http://datastore.example.com",
            folder_id="folder_id",
        )
    kwargs = recursive_call.call_args.kwargs
    assert kwargs["datastore_url"] == "http://datastore.example.com"
    assert kwargs["folder_id"] == "folder_id"
    assert kwargs["jobs"] == 2
    assert kwargs["new_dataset_name"] == "new_name"
