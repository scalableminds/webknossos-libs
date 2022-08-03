from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import gmtime, strftime
from typing import Any, Dict, List, Tuple, Union
from zipfile import ZipFile

import httpx
import numpy as np
import pytest
from tifffile import TiffFile

import webknossos as wk

from .constants import TESTDATA_DIR

pytestmark = [pytest.mark.block_network(allowed_hosts=[".*"])]


def test_compare_tifffile(tmp_path: Path) -> None:
    ds = wk.Dataset(tmp_path, (1, 1, 1))
    l = ds.add_layer_from_images(
        "testdata/tiff/test.*.tiff",
        layer_name="compare_tifffile",
        compress=True,
        category="segmentation",
    )
    data = l.get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with TiffFile("testdata/tiff/test.0000.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        assert np.array_equal(data[:, :, z_index], comparison_slice)


REPO_IMAGES_ARGS: List[
    Tuple[Union[str, List[Path]], Dict[str, Any], str, int, Tuple[int, int, int]]
] = [
    (
        "testdata/tiff/test.*.tiff",
        {"category": "segmentation"},
        "uint8",
        1,
        (265, 265, 257),
    ),
    (
        [
            TESTDATA_DIR / "tiff" / "test.0000.tiff",
            TESTDATA_DIR / "tiff" / "test.0001.tiff",
            TESTDATA_DIR / "tiff" / "test.0002.tiff",
        ],
        {},
        "uint8",
        1,
        (265, 265, 3),
    ),
    (
        "testdata/rgb_tiff/test_rgb.tif",
        {"mag": 2},
        "uint8",
        3,
        (64, 64, 2),
    ),
    (
        "testdata/rgb_tiff/test_rgb.tif",
        {"mag": 2, "channel": 1},
        "uint8",
        1,
        (64, 64, 2),
    ),
    (
        "testdata/temca2/*/*/*.jpg",
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        1,
        (1024, 1024, 12),
    ),
    (
        "testdata/tiff_with_different_dimensions/*",
        {"flip_y": True},
        "uint8",
        1,
        (2970, 2521, 3),
    ),
    ("testdata/various_tiff_formats/test_CS.tif", {}, "uint8", 3, (128, 128, 320)),
    ("testdata/various_tiff_formats/test_C.tif", {}, "uint8", 1, (128, 128, 320)),
    ("testdata/various_tiff_formats/test_I.tif", {}, "uint32", 1, (64, 128, 64)),
    ("testdata/various_tiff_formats/test_S.tif", {}, "uint16", 3, (128, 128, 64)),
]


@pytest.mark.parametrize("path, kwargs, dtype, num_channels, size", REPO_IMAGES_ARGS)
def test_repo_images(
    tmp_path: Path,
    path: str,
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
) -> wk.Dataset:
    with wk.utils.get_executor_for_args(None) as executor:
        ds = wk.Dataset(tmp_path, (1, 1, 1))
        layer_name = "__".join(
            (path if isinstance(path, str) else str(path[0])).split("/")[1:]
        )
        l = ds.add_layer_from_images(
            path,
            layer_name=layer_name,
            compress=True,
            executor=executor,
            **kwargs,
        )
        assert l.dtype_per_channel == np.dtype(dtype)
        assert l.num_channels == num_channels
        assert l.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        if isinstance(l, wk.SegmentationLayer):
            assert l.largest_segment_id > 0
    return ds


def download_and_unpack(url: str, out_path: Path) -> None:
    with NamedTemporaryFile() as download_file:
        with httpx.stream("GET", url) as response:
            total = int(response.headers["Content-Length"])

            with wk.utils.get_rich_progress() as progress:
                download_task = progress.add_task("Download Image Data", total=total)
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        download_task, completed=response.num_bytes_downloaded
                    )
        with ZipFile(download_file, "r") as zip_file:
            zip_file.extractall(out_path)


BIOFORMATS_ARGS = [
    (
        "https://samples.scif.io/wtembryo.zip",
        "wtembryo.mov",
        {},
        "uint8",
        3,
        (320, 240, 108),
    ),
    (
        "https://samples.scif.io/wtembryo.zip",
        "wtembryo.mov",
        {"timepoint": 50, "swap_xy": True},
        "uint8",
        3,
        (240, 320, 1),
    ),
    (
        "https://samples.scif.io/HEART.zip",
        "HEART.SEQ",
        {"flip_z": True},
        "uint8",
        1,
        (512, 512, 30),
    ),
    (
        "https://samples.scif.io/sdub.zip",
        "sdub*.pic",
        {"timepoint": 3},
        "uint8",
        1,
        (192, 128, 9),
    ),
    (
        "https://samples.scif.io/test-avi.zip",
        "t1-rendering.avi",
        {},
        "uint8",
        3,
        (206, 218, 36),
    ),
]


@pytest.mark.parametrize(
    "url, filename, kwargs, dtype, num_channels, size", BIOFORMATS_ARGS
)
def test_bioformats(
    tmp_path: Path,
    url: str,
    filename: str,
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
) -> wk.Dataset:
    unzip_path = tmp_path / "unzip"
    download_and_unpack(url, unzip_path)
    ds = wk.Dataset(tmp_path / "ds", (1, 1, 1))
    with wk.utils.get_executor_for_args(None) as executor:
        l = ds.add_layer_from_images(
            str(unzip_path / filename),
            layer_name=filename,
            compress=True,
            executor=executor,
            use_bioformats=True,
            **kwargs,
        )
        assert l.dtype_per_channel == np.dtype(dtype)
        assert l.num_channels == num_channels
        assert l.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
    return ds


TEST_IMAGES_ARGS = [
    (
        "https://samples.scif.io/test-gif.zip",
        "scifio-test.gif",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-jpeg2000.zip",
        "scifio-test.jp2",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-jpg.zip",
        "scifio-test.jpg",
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-png.zip",
        "scifio-test.png",
        {"flip_y": True},
        "uint8",
        3,
        (500, 500, 1),
    ),
]


@pytest.mark.parametrize(
    "url, filename, kwargs, dtype, num_channels, size", TEST_IMAGES_ARGS
)
def test_test_images(
    tmp_path: Path,
    url: str,
    filename: str,
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
) -> wk.Dataset:
    unzip_path = tmp_path / "unzip"
    download_and_unpack(url, unzip_path)
    ds = wk.Dataset(tmp_path / "ds", (1, 1, 1))
    with wk.utils.get_executor_for_args(None) as executor:
        l_bio = ds.add_layer_from_images(
            str(unzip_path / filename),
            layer_name="bioformats_" + filename,
            compress=True,
            executor=executor,
            use_bioformats=True,
            **kwargs,
        )
        assert l_bio.dtype_per_channel == np.dtype(dtype)
        assert l_bio.num_channels == num_channels
        assert l_bio.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        l_normal = ds.add_layer_from_images(
            str(unzip_path / filename),
            layer_name="normal_" + filename,
            compress=True,
            executor=executor,
            **kwargs,
        )
        assert l_normal.dtype_per_channel == np.dtype(dtype)
        assert l_normal.num_channels == num_channels
        assert l_normal.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        assert np.array_equal(
            l_bio.get_finest_mag().read(), l_normal.get_finest_mag().read()
        )
    return ds


if __name__ == "__main__":
    time = lambda: strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    for repo_images_args in REPO_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            image_path = repo_images_args[0]
            if isinstance(image_path, list):
                image_path = str(image_path[0])
            name = "".join(filter(str.isalnum, image_path))
            print(*repo_images_args)
            print(
                test_repo_images(Path(tempdir), *repo_images_args)
                .upload(f"test_repo_images_{name}_{time()}")
                .url
            )

    for bioformats_args in BIOFORMATS_ARGS:
        with TemporaryDirectory() as tempdir:
            name = "".join(filter(str.isalnum, bioformats_args[1]))
            print(*bioformats_args)
            print(
                test_bioformats(Path(tempdir), *bioformats_args)
                .upload(f"test_bioformats_{name}_{time()}")
                .url
            )

    for test_images_args in TEST_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            name = "".join(filter(str.isalnum, test_images_args[1]))
            print(*test_images_args)
            print(
                test_test_images(Path(tempdir), *test_images_args)
                .upload(f"test_test_images_{name}_{time()}")
                .url
            )
