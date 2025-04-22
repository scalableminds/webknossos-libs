"""This module contains tests for the WEBKNOSSOS CLI."""

import json
import os
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from tests.constants import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    use_minio,
)
from webknossos import BoundingBox, DataFormat, Dataset, Mag
from webknossos.cli.export_as_tiff import _make_tiff_name
from webknossos.cli.main import app
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.defaults import DEFAULT_CHUNK_SHAPE

runner = CliRunner()


@contextmanager
def tmp_cwd() -> Iterator[None]:
    """Creates a temporary working directory to test side effects."""

    prev_cwd = os.getcwd()
    with TemporaryDirectory() as new_cwd:
        os.chdir(new_cwd)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


@pytest.fixture(autouse=True, scope="module")
def minio_docker() -> Iterator[None]:
    with use_minio():
        yield


def check_call(*args: str | int | Path) -> None:
    try:
        subprocess.check_call([str(a) for a in args])
    except subprocess.CalledProcessError as err:
        print(f"Process failed with exit code {err.returncode}: `{args}`")
        raise err


def _tiff_cubing(out_path: Path, data_format: DataFormat) -> None:
    in_path = TESTDATA_DIR / "tiff"

    check_call(
        "webknossos",
        "convert",
        "--jobs",
        2,
        "--voxel-size",
        "11.24,11.24,25",
        "--data-format",
        str(data_format),
        in_path,
        out_path,
    )

    assert (out_path / "tiff").exists()
    assert (out_path / "tiff" / "1").exists()


def test_tiff_cubing_zarr_s3() -> None:
    """Tests zarr support when performing tiff cubing."""

    out_path = REMOTE_TESTOUTPUT_DIR / "tiff_cubing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_ROOT_PASSWORD
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ROOT_USER
    os.environ["S3_ENDPOINT_URL"] = f"http://localhost:{MINIO_PORT}"

    random.seed(1)
    _tiff_cubing(out_path, DataFormat.Zarr3)

    assert (out_path / "tiff" / "1" / "zarr.json").exists()

    json_a = json.loads((out_path / PROPERTIES_FILE_NAME).read_bytes())
    json_fixture = json.loads(
        (TESTDATA_DIR / "tiff" / "datasource-properties.zarr-fixture.json").read_bytes()
    )
    del json_a["id"]
    del json_fixture["id"]
    assert json_a == json_fixture


def test_main() -> None:
    """Tests the functionality of the webknossos command."""

    result_without_args = runner.invoke(app, [])
    assert result_without_args.exit_code == 0


def test_check_equality() -> None:
    """Tests the functionality of check_equality subcommand."""

    result_without_args = runner.invoke(app, ["check-equality"])
    assert result_without_args.exit_code == 2

    result = runner.invoke(
        app,
        [
            "check-equality",
            str(TESTDATA_DIR / "simple_wkw_dataset"),
            str(TESTDATA_DIR / "simple_wkw_dataset"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (
        f"The datasets {str(TESTDATA_DIR / 'simple_wkw_dataset')} and "
        f"{str(TESTDATA_DIR / 'simple_wkw_dataset')} are equal"
        in result.stdout.replace("\n", "")
    )


def test_copy_dataset(tmp_path: Path) -> None:
    """Tests the functionality of copy_dataset subcommand."""

    result_without_args = runner.invoke(app, ["copy-dataset"])
    assert result_without_args.exit_code == 2

    result = runner.invoke(
        app,
        [
            "copy-dataset",
            str(TESTDATA_DIR / "simple_wkw_dataset"),
            str(tmp_path / "simple_wkw_dataset"),
            "--data-format",
            "zarr3",
        ],
    )
    assert result.exit_code == 0
    # verify that data is
    target_ds = Dataset.open(tmp_path / "simple_wkw_dataset")
    target_layer = target_ds.get_layer("color")
    assert target_layer.data_format == DataFormat.Zarr3
    assert Mag(1) in target_layer.mags

    result = runner.invoke(
        app,
        [
            "copy-dataset",
            str(TESTDATA_DIR / "simple_wkw_dataset"),
            str(tmp_path / "simple_wkw_dataset"),
            "--data-format",
            "zarr3",
            "--exists-ok",
        ],
    )
    assert result.exit_code == 0


def test_check_not_equal() -> None:
    """Tests that the check_equality subcommand detects differing datasets."""

    with tmp_cwd():
        tmp_path = Path("testdata") / "simple_wkw_dataset"

        copytree(TESTDATA_DIR / "simple_wkw_dataset", tmp_path)
        dataset = Dataset.open(tmp_path)
        finest_mag = dataset.get_layer("color").get_finest_mag().read()
        finest_mag[1, 5, 7, 4] += 1
        dataset.get_layer("color").get_finest_mag().write(finest_mag)

        result = runner.invoke(
            app,
            [
                "check-equality",
                str(TESTDATA_DIR / "simple_wkw_dataset"),
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1, result.stdout
        assert (
            f"The datasets {str(TESTDATA_DIR / 'simple_wkw_dataset')} and "
            f"{str(TESTDATA_DIR / 'simple_wkw_dataset')} are equal"
            not in result.stdout.replace("\n", "")
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compress() -> None:
    """Tests the functionality of compress subcommand."""

    result_without_args = runner.invoke(app, ["compress"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = TESTDATA_DIR / "simple_wkw_dataset"
        copytree(wkw_path, Path("testdata") / "simple_wkw_dataset")

        result = runner.invoke(app, ["compress", "testdata/simple_wkw_dataset"])

        assert result.exit_code == 0, result.stdout


def test_compress_with_args() -> None:
    """Tests the functionality of compress subcommand."""

    with tmp_cwd():
        wkw_path = TESTDATA_DIR / "simple_wkw_dataset"
        copytree(wkw_path, Path("testdata") / "simple_wkw_dataset")

        result = runner.invoke(
            app,
            [
                "compress",
                "--layer-name",
                "color",
                "--mag",
                "1",
                "testdata/simple_wkw_dataset",
            ],
        )

        assert result.exit_code == 0, result.stdout

        result_with_wrong_mag = runner.invoke(
            app,
            [
                "compress",
                "--layer-name",
                "color",
                "--mag",
                "2",
                "testdata/simple_wkw_dataset",
            ],
        )

        assert result_with_wrong_mag.exit_code == 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_convert() -> None:
    """Tests the functionality of convert subcommand."""

    result_without_args = runner.invoke(app, ["convert"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        origin_path = TESTDATA_DIR / "tiff"
        wkw_path = Path("wkw_from_tiff_simple")

        result = runner.invoke(
            app,
            [
                "convert",
                "--voxel-size",
                "11.0,11.0,11.0",
                str(origin_path),
                str(wkw_path),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (wkw_path / PROPERTIES_FILE_NAME).exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_convert_single_file() -> None:
    """Tests the functionality of convert subcommand when given single file instead of directory."""

    with tmp_cwd():
        origin_path = TESTDATA_DIR / "tiff" / "test.0000.tiff"
        wkw_path = Path("wkw_from_tiff_single_file")

        result = runner.invoke(
            app,
            [
                "convert",
                "--voxel-size",
                "11.0,11.0,11.0",
                str(origin_path),
                str(wkw_path),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (wkw_path / PROPERTIES_FILE_NAME).exists()


def test_convert_with_all_params() -> None:
    """Tests the functionality of convert subcommand."""

    with tmp_cwd():
        origin_path = TESTDATA_DIR / "tiff_with_different_shapes"
        wkw_path = Path(f"wkw_from_{origin_path.name}")
        with pytest.warns(UserWarning, match="Some images are larger than expected,"):
            result = runner.invoke(
                app,
                [
                    "convert",
                    "--voxel-size",
                    "11.0,11.0,11.0",
                    "--data-format",
                    "wkw",
                    "--name",
                    "wkw_from_tiff",
                    "--compress",
                    str(origin_path),
                    str(wkw_path),
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert (wkw_path / PROPERTIES_FILE_NAME).exists()


@pytest.mark.use_proxay
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000/datasets/Organization_X/l4_sample/",
        "http://localhost:9000/datasets/Organization_X/l4_sample/view#512,512,16,0,1.3",
    ],
)
def test_download_dataset(url: str) -> None:
    """Tests the functionality of download subcommand."""

    result = runner.invoke(app, ["download"])
    assert result.exit_code == 2, result.stdout

    with tmp_cwd():
        result = runner.invoke(
            app,
            [
                "download",
                "--bbox",
                "0,0,0,5,5,5",
                "--mag",
                "8-8-1",
                "--url",
                url,
                "testoutput/",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (Path("testoutput") / PROPERTIES_FILE_NAME).exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_downsample_and_upsample() -> None:
    """Tests the functionality of downsample subcommand."""

    result_without_args = runner.invoke(app, ["downsample"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = Path("simple_wkw_dataset")
        copytree(TESTDATA_DIR / wkw_path, wkw_path)

        result_downsample = runner.invoke(app, ["downsample", str(wkw_path)])

        assert result_downsample.exit_code == 0
        assert (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()
        assert (wkw_path / "color" / "2" / "z0" / "y0" / "x0.wkw").exists()
        assert (wkw_path / "color" / "4" / "z0" / "y0" / "x0.wkw").exists()

        Dataset.open(wkw_path).get_layer("color").delete_mag(1)

        assert not (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()

        result_upsample = runner.invoke(
            app, ["upsample", "--from-mag", "2", str(wkw_path)]
        )

        assert result_upsample.exit_code == 0
        assert (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()


def test_upload() -> None:
    """Tests the functionality of upload subcommand."""

    result_without_args = runner.invoke(app, ["upload"])
    assert result_without_args.exit_code == 2


def test_export_tiff_stack(tmp_path: Path) -> None:
    """Tests export of a tiff stack."""

    source_path = TESTDATA_DIR / "simple_wkw_dataset"
    destination_path = tmp_path / "simple_wkw_dataset_tiff"
    bbox = BoundingBox((4, 4, 10), (20, 20, 14))

    result = runner.invoke(
        app,
        [
            "export-as-tiff",
            "--layer-name",
            "color",
            "--name",
            "test_export",
            "--bbox",
            "4,4,10,20,20,14",
            "--mag",
            "1",
            str(source_path),
            str(destination_path),
        ],
    )

    assert result.exit_code == 0, result.stdout

    test_mag_view = Dataset.open(source_path).get_layer("color").get_mag("1")

    for data_slice_index in range(bbox.size.z):
        slice_bbox = BoundingBox(
            (bbox.topleft.x, bbox.topleft.y, bbox.topleft.z + data_slice_index),
            (bbox.size.x, bbox.size.y, 1),
        )
        tiff_path = destination_path / _make_tiff_name(
            "test_export", data_slice_index + 1
        )

        assert tiff_path.is_file(), f"Expected a tiff to be written at: {tiff_path}."

        test_image = np.array(Image.open(tiff_path)).T

        correct_image = test_mag_view.read(
            absolute_offset=slice_bbox.topleft, size=slice_bbox.size
        )
        correct_image = np.squeeze(correct_image)

        assert np.array_equal(correct_image, test_image), (
            f"The tiff file {tiff_path} that was written is not "
            f"equal to the original wkw_file."
        )


def test_export_tiff_stack_tile_size(tmp_path: Path) -> None:
    """Tests the tile size support of exporting a tiff stack."""

    source_path = TESTDATA_DIR / "simple_wkw_dataset"
    destination_path = tmp_path / "simple_wkw_dataset_tile_size"
    bbox = BoundingBox((0, 0, 0), (24, 24, 5))

    result = runner.invoke(
        app,
        [
            "export-as-tiff",
            "--layer-name",
            "color",
            "--name",
            "test_export",
            "--bbox",
            bbox.to_csv(),
            "--mag",
            "1",
            "--tile-size",
            "17,17",
            str(source_path),
            str(destination_path),
        ],
    )

    assert result.exit_code == 0, result.stdout

    tile_bbox = BoundingBox(bbox.topleft, (17, 17, 1))
    test_mag_view = Dataset.open(source_path).get_layer("color").get_mag("1")

    for data_slice_index in range(bbox.size.z):
        for y_tile_index in range(ceil(bbox.size.y / tile_bbox.size.y)):
            for x_tile_index in range(ceil(bbox.size.x / tile_bbox.size.x)):
                tiff_path = (
                    destination_path
                    / f"{data_slice_index + 1}"
                    / f"{y_tile_index + 1}"
                    / f"{x_tile_index + 1}.tiff"
                )

                assert tiff_path.is_file(), (
                    f"Expected a tiff to be written at: {tiff_path}."
                )

                test_image = np.array(Image.open(tiff_path)).T

                correct_image = test_mag_view.read(
                    absolute_offset=(
                        tile_bbox.topleft.x + tile_bbox.size.x * x_tile_index,
                        tile_bbox.topleft.y + tile_bbox.size.y * y_tile_index,
                        tile_bbox.topleft.z + data_slice_index,
                    ),
                    size=tile_bbox.size,
                )

                correct_image = np.squeeze(correct_image)

                assert np.array_equal(correct_image, test_image), (
                    f"The tiff file {tiff_path} that was written "
                    f"is not equal to the original wkw_file."
                )


def test_export_tiff_stack_tiles_per_dimension(tmp_path: Path) -> None:
    """Tests the tiles per dimension support when exporting a tiff stack."""

    source_path = TESTDATA_DIR / "simple_wkw_dataset"
    destination_path = tmp_path / "simple_wkw_dataset_tiles_per_dimension"
    bbox = BoundingBox((0, 0, 0), (24, 24, 5))

    result = runner.invoke(
        app,
        [
            "export-as-tiff",
            "--layer-name",
            "color",
            "--name",
            "test_export",
            "--bbox",
            bbox.to_csv(),
            "--mag",
            "1",
            "--tiles-per-dimension",
            "3,3",
            str(source_path),
            str(destination_path),
        ],
    )

    assert result.exit_code == 0, result.stdout

    tile_bbox = BoundingBox(bbox.topleft, (8, 8, 1))
    test_mag_view = Dataset.open(source_path).get_layer("color").get_mag("1")

    for data_slice_index in range(bbox.size.z):
        for y_tile_index in range(ceil(bbox.size.y / tile_bbox.size.y)):
            for x_tile_index in range(ceil(tile_bbox.size.x / tile_bbox.size.x)):
                tiff_path = (
                    destination_path
                    / f"{data_slice_index + 1}"
                    / f"{y_tile_index + 1}"
                    / f"{x_tile_index + 1}.tiff"
                )

                assert tiff_path.is_file(), (
                    f"Expected a tiff to be written at: {tiff_path}."
                )

                test_image = np.array(Image.open(tiff_path)).T

                correct_image = test_mag_view.read(
                    absolute_offset=(
                        tile_bbox.topleft.x + tile_bbox.size.x * x_tile_index,
                        tile_bbox.topleft.y + tile_bbox.size.y * y_tile_index,
                        tile_bbox.topleft.z + data_slice_index,
                    ),
                    size=tile_bbox.size,
                )

                correct_image = np.squeeze(correct_image)

                assert np.array_equal(correct_image, test_image), (
                    f"The tiff file {tiff_path} that was written "
                    f"is not equal to the original wkw_file."
                )


@pytest.mark.parametrize("use_compression", [True, False])
def test_merge_fallback_no_fallback_layer(
    tmp_path: Path, use_compression: bool
) -> None:
    from zipfile import ZIP_DEFLATED, ZipFile
    from zlib import Z_BEST_SPEED

    import webknossos.annotation
    from webknossos import SEGMENTATION_CATEGORY, Annotation, Skeleton

    fallback_layer_data = np.ones((64, 64, 64), dtype=np.uint8)

    fallback_mag = (
        Dataset(tmp_path / "fallback_dataset", (11.24, 11.24, 25))
        .add_layer(
            "fallback_layer",
            SEGMENTATION_CATEGORY,
            dtype_per_channel=fallback_layer_data.dtype,
            data_format=DataFormat.WKW,
        )
        .add_mag(
            1,
            chunk_shape=DEFAULT_CHUNK_SHAPE,
            shard_shape=DEFAULT_CHUNK_SHAPE,
            compress=use_compression,
        )
    )

    fallback_mag.write(
        absolute_offset=(0,) * 3, data=fallback_layer_data, allow_resize=True
    )

    annotation_zip_path = tmp_path / "annotation.zip"
    annotation_data = np.ones((32, 32, 32), dtype=fallback_layer_data.dtype) * 2
    voxel_size = (11.24, 11.24, 25)

    topleft = (32,) * 3

    with TemporaryDirectory(dir=tmp_path) as tmp_dir:
        tmp_ds_dir = Path(tmp_dir)
        tmp_dataset = Dataset(tmp_ds_dir / "tmp_dataset", voxel_size)

        largest_segment_id = int(annotation_data.max())

        tmp_layer = tmp_dataset.add_layer(
            "Volume",
            SEGMENTATION_CATEGORY,
            data_format=DataFormat.WKW,
            dtype_per_channel=annotation_data.dtype,
            largest_segment_id=largest_segment_id,
        )

        mag1 = tmp_layer.add_mag(
            1,
            chunk_shape=DEFAULT_CHUNK_SHAPE,
            shard_shape=DEFAULT_CHUNK_SHAPE,
            compress=True,
        )

        mag1.write(absolute_offset=topleft, data=annotation_data, allow_resize=True)

        volume_layer_zip = tmp_ds_dir / "data_Volume.zip"

        with ZipFile(
            volume_layer_zip,
            mode="x",
            compression=ZIP_DEFLATED,
            compresslevel=Z_BEST_SPEED,
        ) as zf:
            for dirname, _, files in os.walk(str(tmp_layer.path)):
                arcname = str(Path(dirname).relative_to(tmp_layer.path))
                for filename in files:
                    if filename.endswith(".wkw"):
                        zf.write(
                            os.path.join(dirname, filename),
                            os.path.join(arcname, filename),
                        )

        annotation = Annotation(
            name="test_annotation",
            skeleton=Skeleton(
                voxel_size=tmp_dataset.voxel_size,
                dataset_name=fallback_mag.layer.dataset.name,
            ),
        )

        annotation._volume_layers = [
            webknossos.annotation._VolumeLayer(  # type: ignore
                id=0,
                name=tmp_layer.name,
                fallback_layer_name=fallback_mag.layer.name,
                zip=volume_layer_zip,
                segments={},
                data_format=DataFormat.WKW,
                largest_segment_id=largest_segment_id,
            ),
        ]

        annotation.save(annotation_zip_path)

    target_dataset_path = tmp_path / "merged_dataset"

    result = runner.invoke(
        app,
        [
            "merge-fallback",
            str(target_dataset_path),
            str(annotation_zip_path),
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout

    expected_data = fallback_layer_data
    expected_data[
        tuple(slice(t, t + s) for t, s in zip(topleft, annotation_data.shape))
    ] = annotation_data

    merged_data = (
        Dataset.open(target_dataset_path)
        .get_layer(fallback_mag.layer.name)
        .get_mag(1)
        .read()
        .squeeze(0)
    )

    np.testing.assert_array_equal(merged_data, expected_data)
