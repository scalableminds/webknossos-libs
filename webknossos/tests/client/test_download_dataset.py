from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from upath import UPath

from webknossos import BoundingBox, Mag
from webknossos.client._download_dataset import download_dataset
from webknossos.dataset_properties import VoxelSize


def _api_layer(name: str, mags: list[Mag]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        category="color",
        dtype_np=np.dtype("uint8"),
        bounding_box=SimpleNamespace(size=SimpleNamespace(c=1)),
        default_view_configuration=None,
        mags=[SimpleNamespace(mag=mag) for mag in mags],
    )


def test_download_dataset_uses_mags_of_each_layer(tmp_upath: UPath) -> None:
    api_dataset = MagicMock()
    api_dataset.name = "ds"
    api_dataset.data_source.scale = VoxelSize((1, 1, 1))
    api_dataset.data_source.data_layers = [
        _api_layer("layer1", [Mag(1)]),
        _api_layer("layer2", [Mag(1), Mag(2)]),
    ]

    def get_raw_data(**kwargs: Any) -> tuple[bytes, str]:
        num_bytes = kwargs["width"] * kwargs["height"] * kwargs["depth"]
        return bytes(num_bytes), "[]"

    context = MagicMock()
    context.api_client.dataset_info.return_value = api_dataset
    datastore_client = context.get_datastore_api_client.return_value
    datastore_client.dataset_get_raw_data.side_effect = get_raw_data

    with patch(
        "webknossos.client._download_dataset._get_context", return_value=context
    ):
        dataset = download_dataset(
            "dataset_id",
            bbox=BoundingBox((0, 0, 0), (4, 4, 4)),
            path=tmp_upath / "ds",
        )

    assert set(dataset.get_layer("layer1").mags) == {Mag(1)}
    assert set(dataset.get_layer("layer2").mags) == {Mag(1), Mag(2)}
