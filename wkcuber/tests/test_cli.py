import json
import subprocess
from pathlib import Path
from shutil import unpack_archive
from typing import Union

import numpy as np
import pytest
import wkw
from webknossos import Dataset, Mag
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.utils import copytree, rmtree

from .constants import TESTDATA_DIR


def check_call(*args: Union[str, int, Path]) -> None:
    try:
        subprocess.check_call([str(a) for a in args])
    except subprocess.CalledProcessError as e:
        print(f"Process failed with exit code {e.returncode}: `{args}`")
        raise e


def compare_wkw(path_a: Path, path_b: Path) -> None:
    with wkw.Dataset.open(str(path_a)) as ds_a, wkw.Dataset.open(str(path_b)) as ds_b:
        assert ds_a.header.version == ds_b.header.version
        assert ds_a.header.block_len == ds_b.header.block_len
        assert ds_a.header.file_len == ds_b.header.file_len
        assert ds_a.header.voxel_type == ds_b.header.voxel_type
        assert ds_a.header.num_channels == ds_b.header.num_channels
        data_a = ds_a.read((0, 0, 0), (1024, 1024, 1024))
        data_b = ds_b.read((0, 0, 0), (1024, 1024, 1024))
        assert np.all(data_a == data_b)


def count_wkw_files(mag_path: Path) -> int:
    return len(list(mag_path.glob("**/x*.wkw")))


def test_tiff_cubing(tmp_path: Path) -> None:
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
        32,
        "--scale",
        "1,1,1",
        in_path,
        tmp_path,
    )

    assert (tmp_path / "color").exists()
    assert (tmp_path / "color" / "1").exists()
    assert count_wkw_files(tmp_path / "color" / "1") == 1

    check_call(
        "python",
        "-m",
        "wkcuber.metadata",
        "--name",
        "test_dataset",
        "--scale",
        "11.24,11.24,25",
        tmp_path,
    )
    assert (tmp_path / PROPERTIES_FILE_NAME).exists()
    with (tmp_path / PROPERTIES_FILE_NAME).open("r") as a, (
        in_path / "datasource-properties.fixture.json"
    ).open("r") as b:
        assert json.load(a) == json.load(b)


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

    compare_wkw(tmp_path / "color" / "2", tiff_mag_2_reference_path / "color" / "2")


def test_upsampling(
    sample_wkw_path: Path, tmp_path: Path, tiff_mag_2_reference_path: Path
) -> None:
    copytree(sample_wkw_path, tmp_path)

    Dataset.open(tmp_path).get_layer("color").delete_mag("1")

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

    Dataset.open(tmp_path).get_layer("color").delete_mag("2")

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

    compare_wkw(
        tmp_path / "color" / "2",
        tiff_mag_2_reference_path / "color" / "2",
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

    # Generate metadata
    check_call(
        "python",
        "-m",
        "wkcuber.metadata",
        "--name",
        "great_dataset",
        "--scale",
        "11.24,11.24,25",
        out_path,
    )

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
        "--scale",
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
        "--scale",
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
        "--scale",
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
        "--scale",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--wkw_file_len",
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
        "--scale",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--wkw_file_len",
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
        "--scale",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--wkw_file_len",
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
        "--scale",
        "1,1,1",
        "--name",
        "awesome_data",
        "--no_compress",
        "--wkw_file_len",
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
            "--scale",
            "1,1,1",
            "--prefer_channels",
            "--name",
            "awesome_data",
            "--no_compress",
            "--wkw_file_len",
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
        "--scale",
        "1,1,1",
        "--channel_index",
        3,
        "--sample_index",
        2,
        "--name",
        "awesome_data",
        "--no_compress",
        "--wkw_file_len",
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
        "--scale",
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
        "--scale",
        "11.24,11.24,25",
        in_path,
        out_path,
    )

    assert (out_path / "color").is_dir()
    assert (out_path / "color" / "1").is_dir()
    assert count_wkw_files(out_path / "color" / "1") == 1
    assert (out_path / PROPERTIES_FILE_NAME).exists()
