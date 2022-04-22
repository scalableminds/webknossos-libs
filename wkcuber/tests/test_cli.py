import json
import subprocess
import sys
from os import environ
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from upath import UPath
from webknossos import DataFormat, Dataset, Mag
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.utils import copytree, rmtree

from .constants import TESTDATA_DIR
import subprocess
import shlex


def check_call(*args: Union[str, int, Path]) -> None:
    try:
        subprocess.check_call([str(a) for a in args])
    except subprocess.CalledProcessError as e:
        print(f"Process failed with exit code {e.returncode}: `{args}`")
        raise e


def count_wkw_files(mag_path: Path) -> int:
    return len(list(mag_path.glob("**/x*.wkw")))


MINIO_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
MINIO_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
MINIO_PORT = "8000"


@pytest.fixture(scope="module")
def remote_testoutput_path() -> UPath:
    """Minio is an S3 clone and is used as local test server"""
    container_name = "minio"
    cmd = (
        "docker run"
        f" -p {MINIO_PORT}:9000"
        f" -e MINIO_ROOT_USER={MINIO_ROOT_USER}"
        f" -e MINIO_ROOT_PASSWORD={MINIO_ROOT_PASSWORD}"
        f" --name {container_name}"
        " --rm"
        " -d"
        " minio/minio server /data"
    )
    subprocess.check_output(shlex.split(cmd))
    remote_path = UPath(
        "s3://testoutput",
        key=MINIO_ROOT_USER,
        secret=MINIO_ROOT_PASSWORD,
        client_kwargs={"endpoint_url": f"http://localhost:{MINIO_PORT}"},
    )
    remote_path.fs.mkdirs("testoutput", exist_ok=True)
    try:
        yield remote_path
    finally:
        subprocess.check_output(["docker", "stop", container_name])


def _tiff_cubing(
    out_path: Path, data_format: DataFormat, chunks_per_shard: int
) -> None:
    in_path = TESTDATA_DIR / "tiff"

    check_call(
        "python",
        "-m",
        "wkcuber.cubing",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--chunks_per_shard",
        chunks_per_shard,
        "--voxel_size",
        "11.24,11.24,25",
        "--data_format",
        str(data_format),
        in_path,
        out_path,
    )

    assert (out_path / "color").exists()
    assert (out_path / "color" / "1").exists()


def test_tiff_cubing(tmp_path: Path) -> None:
    _tiff_cubing(tmp_path, DataFormat.WKW, 32)

    assert count_wkw_files(tmp_path / "color" / "1") == 1

    assert (tmp_path / PROPERTIES_FILE_NAME).exists()
    with (tmp_path / PROPERTIES_FILE_NAME).open("r") as a, (
        TESTDATA_DIR / "tiff" / "datasource-properties.wkw-fixture.json"
    ).open("r") as fixture_path:
        json_a = json.load(a)
        json_fixture = json.load(fixture_path)
        del json_a["id"]
        del json_fixture["id"]
        assert json_a == json_fixture


@pytest.mark.skipif(
    not sys.platform == "linux",
    reason="Only run this test on Linux, because it requires a running `minio` docker container.",
)
def test_tiff_cubing_zarr_s3(remote_testoutput_path: UPath) -> None:
    out_path = remote_testoutput_path / "tiff_cubing"
    environ["AWS_SECRET_ACCESS_KEY"] = MINIO_ROOT_PASSWORD
    environ["AWS_ACCESS_KEY_ID"] = MINIO_ROOT_USER
    environ["S3_ENDPOINT_URL"] = f"http://localhost:{MINIO_PORT}"

    _tiff_cubing(out_path, DataFormat.Zarr, 1)

    assert (out_path / "color" / "1" / ".zarray").exists()
    assert (out_path / PROPERTIES_FILE_NAME).exists()

    with (out_path / PROPERTIES_FILE_NAME).open("r") as a, (
        TESTDATA_DIR / "tiff" / "datasource-properties.zarr-fixture.json"
    ).open("r") as fixture:
        json_a = json.load(a)
        json_fixture = json.load(fixture)
        del json_a["id"]
        del json_fixture["id"]
        assert json_a == json_fixture


def test_downsampling(
    sample_wkw_path: Path, tmp_path: Path, tiff_mag_2_reference_path: Path
) -> None:
    copytree(sample_wkw_path, tmp_path)
    Dataset.open(tmp_path).get_layer("color").delete_mag("2")

    check_call(
        "python",
        "-m",
        "wkcuber.downsampling",
        "--jobs",
        2,
        "--max",
        8,
        "--buffer_cube_size",
        128,
        "--layer_name",
        "color",
        "--sampling_mode",
        "isotropic",
        tmp_path,
    )
    assert (tmp_path / "color" / "2").exists()
    assert (tmp_path / "color" / "4").exists()
    assert (tmp_path / "color" / "8").exists()
    assert not (tmp_path / "color" / "16").exists()

    assert count_wkw_files(tmp_path / "color" / "2") == 1
    assert count_wkw_files(tmp_path / "color" / "4") == 1
    assert count_wkw_files(tmp_path / "color" / "8") == 1

    assert (
        Dataset.open(tmp_path)
        .get_layer("color")
        .get_mag("2")
        .content_is_equal(
            Dataset.open(tiff_mag_2_reference_path).get_layer("color").get_mag("2")
        )
    )


def test_upsampling(
    sample_wkw_path: Path, tmp_path: Path, tiff_mag_2_reference_path: Path
) -> None:
    copytree(sample_wkw_path, tmp_path)

    color_layer = Dataset.open(tmp_path).get_layer("color")
    color_layer.delete_mag("1")
    color_layer.bounding_box = color_layer.bounding_box.align_with_mag(
        Mag("2"), ceil=True
    )

    check_call(
        "python",
        "-m",
        "wkcuber.upsampling",
        "--jobs",
        2,
        "--from_mag",
        "2-2-2",
        "--target_mag",
        1,
        "--buffer_cube_size",
        1024,
        "--layer_name",
        "color",
        tmp_path,
    )

    color_layer = Dataset.open(tmp_path).get_layer("color")
    color_layer.delete_mag("2")

    check_call(
        "python",
        "-m",
        "wkcuber.downsampling",
        "--jobs",
        2,
        "--from_mag",
        1,
        "--max",
        2,
        "--sampling_mode",
        "isotropic",
        "--buffer_cube_size",
        256,
        "--layer_name",
        "color",
        "--interpolation_mode",
        "nearest",
        tmp_path,
    )

    assert (Dataset.open(tmp_path).get_layer("color").get_mag("2").bounding_box) == (
        Dataset.open(tiff_mag_2_reference_path)
        .get_layer("color")
        .get_mag("2")
        .bounding_box
    )

    assert (
        Dataset.open(tmp_path)
        .get_layer("color")
        .get_mag("2")
        .content_is_equal(
            Dataset.open(tiff_mag_2_reference_path).get_layer("color").get_mag("2")
        )
    )


def test_anisotropic_downsampling(sample_wkw_path: Path, tmp_path: Path) -> None:
    copytree(sample_wkw_path, tmp_path)

    check_call(
        "python",
        "-m",
        "wkcuber.downsampling",
        "--jobs",
        2,
        "--from",
        1,
        "--max",
        2,
        "--sampling_mode",
        "constant_z",
        "--buffer_cube_size",
        128,
        "--layer_name",
        "color",
        tmp_path,
    )

    check_call(
        "python",
        "-m",
        "wkcuber.downsampling",
        "--jobs",
        2,
        "--from",
        "2-2-1",
        "--max",
        4,
        "--sampling_mode",
        "constant_z",
        "--buffer_cube_size",
        128,
        "--layer_name",
        "color",
        tmp_path,
    )

    assert (tmp_path / "color" / "2-2-1").exists()
    assert (tmp_path / "color" / "4-4-1").exists()
    assert count_wkw_files(tmp_path / "color" / "2-2-1") == 1
    assert count_wkw_files(tmp_path / "color" / "4-4-1") == 1

    check_call(
        "python",
        "-m",
        "wkcuber.downsampling",
        "--jobs",
        2,
        "--from",
        "4-4-1",
        "--max",
        16,
        "--buffer_cube_size",
        128,
        "--layer_name",
        "color",
        tmp_path,
    )

    assert (tmp_path / "color" / "8-8-4").exists()
    assert (tmp_path / "color" / "16-16-8").exists()
    assert count_wkw_files(tmp_path / "color" / "8-8-4") == 1
    assert count_wkw_files(tmp_path / "color" / "16-16-8") == 1


def test_compression_and_verification(sample_wkw_path: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "tiff_compress"
    broken_path = tmp_path / "tiff_compress_broken"

    check_call(
        "python",
        "-m",
        "wkcuber.compress",
        "--jobs",
        2,
        "--layer_name",
        "color",
        sample_wkw_path,
        out_path,
    )

    assert (out_path / "color" / "1").is_dir()
    assert (out_path / "color" / "2").is_dir()

    # Check equality for uncompressed and compressed dataset
    check_call("python", "-m", "wkcuber.check_equality", sample_wkw_path, out_path)

    # Create broken copy of dataset
    copytree(out_path, broken_path)
    rmtree(broken_path / "color" / "1" / "z0" / "y0" / "x0.wkw")

    # Compare original dataset to broken one and expect to determine difference
    with pytest.raises(subprocess.CalledProcessError):
        check_call(
            "python", "-m", "wkcuber.check_equality", sample_wkw_path, broken_path
        )


def test_in_place_compression(sample_wkw_path: Path, tmp_path: Path) -> None:
    copytree(sample_wkw_path, tmp_path)

    check_call(
        "python",
        "-m",
        "wkcuber.compress",
        "--jobs",
        2,
        "--layer_name",
        "color",
        tmp_path,
    )

    assert Dataset.open(tmp_path).get_layer("color").get_mag("1").info.compression_mode
    assert Dataset.open(tmp_path).get_layer("color").get_mag("2").info.compression_mode


def test_tile_cubing(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.tile_cubing",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--voxel_size",
        "1,1,1",
        TESTDATA_DIR / "temca2",
        tmp_path,
    )
    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color" / "1") == 8


def test_simple_tiff_cubing(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        4,
        "--voxel_size",
        "11.24,11.24,25",
        "--name",
        "awesome_data",
        "--sampling_mode",
        "isotropic",
        TESTDATA_DIR / "tiff",
        tmp_path,
    )

    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color" / "1") == 1
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()


def test_simple_tiff_cubing_no_compression(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        4,
        "--no_compress",
        "--voxel_size",
        "11.24,11.24,25",
        "--name",
        "awesome_data",
        "--sampling_mode",
        "isotropic",
        TESTDATA_DIR / "tiff",
        tmp_path,
    )

    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color" / "1") == 1
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()


def test_tiff_formats_cubing_I(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        4,
        "--target_mag",
        2,
        "--voxel_size",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--chunks_per_shard",
        8,
        TESTDATA_DIR / "various_tiff_formats" / "test_I.tif",
        tmp_path,
    )

    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "2").is_dir()
    assert count_wkw_files(tmp_path / "color" / "2") == 1
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()


def test_tiff_formats_cubing_C(tmp_path: Path) -> None:
    # to make output wk_compatible, these files require multiple layer
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        1,
        "--voxel_size",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--chunks_per_shard",
        8,
        TESTDATA_DIR / "various_tiff_formats" / "test_C.tif",
        tmp_path,
    )

    assert (tmp_path / "color_0").is_dir()
    assert (tmp_path / "color_1").is_dir()
    assert (tmp_path / "color_2").is_dir()
    assert (tmp_path / "color_3").is_dir()
    assert (tmp_path / "color_4").is_dir()
    assert (tmp_path / "color_0" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color_0" / "1") == 1
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()


def test_tiff_formats_cubing_S(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        1,
        "--voxel_size",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--chunks_per_shard",
        8,
        TESTDATA_DIR / "various_tiff_formats" / "test_S.tif",
        tmp_path,
    )

    assert (tmp_path / "color_0").is_dir()
    assert (tmp_path / "color_1").is_dir()
    assert (tmp_path / "color_2").is_dir()
    assert (tmp_path / "color_0" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color_0" / "1") == 1
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()


def test_tiff_formats_cubing_CS(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        1,
        "--voxel_size",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--chunks_per_shard",
        8,
        TESTDATA_DIR / "various_tiff_formats" / "test_CS.tif",
        tmp_path,
    )

    assert (tmp_path / "color_0").is_dir()
    assert (tmp_path / "color_1").is_dir()
    assert (tmp_path / "color_2").is_dir()
    assert (tmp_path / "color_3").is_dir()
    assert (tmp_path / "color_4").is_dir()
    assert (tmp_path / "color_0" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color_0" / "1") == 1


def test_tiff_formats_cubing_invalid_C(tmp_path: Path) -> None:
    # test wk incompatible configs, program should fail because force parameter was not given
    with pytest.raises(subprocess.CalledProcessError):
        check_call(
            "python",
            "-m",
            "wkcuber.convert_image_stack_to_wkw",
            "--jobs",
            2,
            "--batch_size",
            8,
            "--layer_name",
            "color",
            "--max_mag",
            1,
            "--voxel_size",
            "1,1,1",
            "--prefer_channels",
            "--name",
            "awesome_data",
            "--no_compress",
            "--chunks_per_shard",
            8,
            TESTDATA_DIR / "various_tiff_formats" / "test_C.tif",
            tmp_path,
        )


def test_tiff_formats_cubing_single_layer_CS(tmp_path: Path) -> None:
    # power user configuration: should only create single layer
    check_call(
        "python",
        "-m",
        "wkcuber.convert_image_stack_to_wkw",
        "--jobs",
        2,
        "--batch_size",
        8,
        "--layer_name",
        "color",
        "--max_mag",
        1,
        "--voxel_size",
        "1,1,1",
        "--channel_index",
        3,
        "--sample_index",
        2,
        "--name",
        "awesome_data",
        "--no_compress",
        "--chunks_per_shard",
        8,
        TESTDATA_DIR / "various_tiff_formats" / "test_CS.tif",
        tmp_path,
    )

    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color" / "1") == 1


def test_convert_knossos(tmp_path: Path) -> None:
    check_call(
        "python",
        "-m",
        "wkcuber.convert_knossos",
        "--jobs",
        2,
        "--dtype",
        "uint8",
        "--layer_name",
        "color",
        "--mag",
        1,
        "--voxel_size",
        "1,1,1",
        TESTDATA_DIR / "knossos" / "color" / "1",
        tmp_path,
    )

    assert (tmp_path / "color").is_dir()
    assert (tmp_path / "color" / "1").is_dir()
    assert count_wkw_files(tmp_path / "color" / "1") == 1


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_convert_raw(tmp_path: Path, dtype: str) -> None:
    in_path = tmp_path / f"data_{dtype}.raw"
    out_path = tmp_path / "raw"

    np.arange(128 ** 3, dtype=dtype).reshape(128, 128, 128).tofile(in_path)

    check_call(
        "python",
        "-m",
        "wkcuber.convert_raw",
        "--layer_name",
        "color",
        "--input_dtype",
        dtype,
        "--shape",
        "128,128,128",
        "--voxel_size",
        "11.24,11.24,25",
        in_path,
        out_path,
    )

    assert (out_path / "color").is_dir()
    assert (out_path / "color" / "1").is_dir()
    assert count_wkw_files(out_path / "color" / "1") == 1
    assert (out_path / PROPERTIES_FILE_NAME).exists()
