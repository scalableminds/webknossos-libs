import os
from pathlib import Path
from shutil import rmtree
from timeit import default_timer

import numpy as np
import pytest
from pytest import fixture

import webknossos as wk

TEST_SIZE = int(os.environ.get("TEST_SIZE", "1024"))

PARTIAL_SELECTION = ((32, 32, 32), (32, 32, 32))

TEST_DATASET = wk.Dataset.open("testdata/l4_sample")
TESTDATA: dict[str, np.ndarray] = {
    "color": TEST_DATASET.get_layer("color")
    .get_mag(1)
    .read(absolute_offset=(3072, 3072, 512), size=(TEST_SIZE, TEST_SIZE, TEST_SIZE))[0],
    "segmentation": TEST_DATASET.get_layer("segmentation")
    .get_mag(1)
    .read(absolute_offset=(3072, 3072, 512), size=(TEST_SIZE, TEST_SIZE, TEST_SIZE))[0],
}


@fixture
def folder() -> Path:
    path = Path("testdata")
    rmtree(path, ignore_errors=True)
    return path


def folder_disk_usage(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.glob("**/*") if f.is_file())


def folder_inodes(folder: Path) -> int:
    return sum(1 for _ in folder.glob("**/*"))


@pytest.mark.parametrize("src_layer_name", ["color", "segmentation"])
@pytest.mark.parametrize(
    "data_format", [wk.DataFormat.WKW, wk.DataFormat.Zarr3, wk.DataFormat.Zarr]
)
def test_perf(
    tmp_path: Path,
    src_layer_name: str,
    data_format: wk.DataFormat,
) -> None:
    print("")
    src_layer = TEST_DATASET.get_layer(src_layer_name)
    testdata = TESTDATA[src_layer_name]
    dataset = wk.Dataset(tmp_path / "l4_sample_perf", src_layer.dataset.voxel_size)
    layer = dataset.add_layer(
        src_layer_name,
        src_layer.category,
        data_format=data_format,
        dtype_per_channel=src_layer.dtype_per_channel,
    )
    mag = layer.add_mag(1, compress=True)

    start = default_timer()
    mag.write(absolute_offset=(0, 0, 0), data=testdata)
    print(f"  {data_format} WRITE {src_layer_name} - {default_timer() - start:.2f}s")

    print(
        f"  {data_format} STORAGE {folder_disk_usage(mag.path.resolve())/1000000:,.2f} MB "
        + f"- {folder_inodes(mag.path.resolve())} inodes"
    )

    start = default_timer()
    readback_data = mag.read(absolute_offset=(0, 0, 0), size=testdata.shape)[0]
    print(f"  {data_format} READ {src_layer_name} - {default_timer() - start:.2f}s")
    assert np.array_equal(readback_data, testdata)

    start = default_timer()
    partial_data = mag.read(
        absolute_offset=PARTIAL_SELECTION[0], size=PARTIAL_SELECTION[1]
    )[0]
    print(
        f"  {data_format} PARTIAL READ {src_layer_name} - {default_timer() - start:.5f}s"
    )
    assert np.array_equal(
        partial_data,
        testdata[
            tuple(
                slice(offset, offset + size) for offset, size in zip(*PARTIAL_SELECTION)
            )
        ],
    )
