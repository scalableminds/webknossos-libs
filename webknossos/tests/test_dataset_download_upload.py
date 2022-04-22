from pathlib import Path
from tempfile import TemporaryDirectory
from time import gmtime, strftime
from typing import Iterator

import numpy as np
import pytest

import webknossos as wk

# pylint: disable=redefined-outer-name

pytestmark = [pytest.mark.with_vcr]


@pytest.fixture(scope="module")
def sample_bbox() -> wk.BoundingBox:
    return wk.BoundingBox((2807, 4352, 1794), (10, 10, 10))


@pytest.fixture(scope="module")
def sample_dataset(sample_bbox: wk.BoundingBox) -> Iterator[wk.Dataset]:
    url = "https://webknossos.org/datasets/scalable_minds/l4_sample_dev"
    with TemporaryDirectory() as temp_dir:
        yield wk.Dataset.download(url, path=Path(temp_dir) / "ds", bbox=sample_bbox)


@pytest.mark.parametrize(
    "url",
    [
        "https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view",
        "https://webknossos.org/datasets/scalable_minds/l4_sample_dev_sharing/view?token=ilDXmfQa2G8e719vb1U9YQ#%7B%22orthogonal%7D",
    ],
)
def test_url_download(
    url: str, tmp_path: Path, sample_dataset: wk.Dataset, sample_bbox: wk.BoundingBox
) -> None:
    ds = wk.Dataset.download(
        url, path=tmp_path / "ds", mags=[wk.Mag(1)], bbox=sample_bbox
    )
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = ds.get_color_layers()[0].get_finest_mag().read()
    assert data.sum() == 122507
    assert np.array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


def test_upload_download_roundtrip(sample_dataset: wk.Dataset, tmp_path: Path) -> None:
    ds_original = sample_dataset
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    url = ds_original.upload(
        new_dataset_name=f"test_upload_download_roundtrip_{time_str}"
    )
    ds_roundtrip = wk.Dataset.download(
        url, path=tmp_path / "ds", layers=["color", "segmentation"]
    )
    assert set(ds_original.get_segmentation_layers()[0].mags.keys()) == set(
        ds_roundtrip.get_segmentation_layers()[0].mags.keys()
    )

    original_config = ds_original.get_layer("color").default_view_configuration
    roundtrip_config = ds_roundtrip.get_layer("color").default_view_configuration
    assert (
        original_config is not None
    ), "default_view_configuration should be defined for original dataset"
    assert (
        roundtrip_config is not None
    ), "default_view_configuration should be defined for roundtrip dataset"
    assert original_config.color == roundtrip_config.color
    assert original_config.intensity_range == roundtrip_config.intensity_range

    data_original = ds_original.get_segmentation_layers()[0].get_finest_mag().read()
    data_roundtrip = ds_roundtrip.get_segmentation_layers()[0].get_finest_mag().read()
    assert np.array_equal(data_original, data_roundtrip)
