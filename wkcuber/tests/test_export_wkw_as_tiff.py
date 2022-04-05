from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image
from webknossos import BoundingBox, Dataset
from wkcuber.export_wkw_as_tiff import _make_tiff_name, run

from .constants import TESTDATA_DIR

DS_NAME = "simple_wkw_dataset"
SOURCE_PATH = TESTDATA_DIR / DS_NAME


def test_export_tiff_stack(tmp_path: Path) -> None:
    destination_path = tmp_path / f"{DS_NAME}_tiff"
    bbox = BoundingBox((4, 4, 10), (20, 20, 14))
    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        str(destination_path),
        "--layer_name",
        "color",
        "--name",
        "test_export",
        "--bbox",
        bbox.to_csv(),
        "--mag",
        "1",
    ]

    run(args_list)

    test_mag_view = Dataset.open(SOURCE_PATH).get_layer("color").get_mag("1")

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
    destination_path = tmp_path / f"{DS_NAME}_tile_size"
    bbox = BoundingBox((0, 0, 0), (24, 24, 5))

    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        str(destination_path),
        "--layer_name",
        "color",
        "--name",
        "test_export",
        "--bbox",
        bbox.to_csv(),
        "--mag",
        "1",
        "--tile_size",
        "17,17",
    ]

    run(args_list)

    tile_bbox = BoundingBox(bbox.topleft, (17, 17, 1))
    test_mag_view = Dataset.open(SOURCE_PATH).get_layer("color").get_mag("1")

    for data_slice_index in range(bbox.size.z):
        for y_tile_index in range(ceil(bbox.size.y / tile_bbox.size.y)):
            for x_tile_index in range(ceil(bbox.size.x / tile_bbox.size.x)):
                tiff_path = (
                    destination_path
                    / f"{data_slice_index + 1}"
                    / f"{y_tile_index + 1}"
                    / f"{x_tile_index + 1}.tiff"
                )

                assert (
                    tiff_path.is_file()
                ), f"Expected a tiff to be written at: {tiff_path}."

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
    destination_path = tmp_path / f"{DS_NAME}_tiles_per_dimension"
    bbox = BoundingBox((0, 0, 0), (24, 24, 5))

    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        str(destination_path),
        "--layer_name",
        "color",
        "--name",
        "test_export",
        "--bbox",
        bbox.to_csv(),
        "--mag",
        "1",
        "--tiles_per_dimension",
        "3,3",
    ]

    run(args_list)

    tile_bbox = BoundingBox(bbox.topleft, (8, 8, 1))
    test_mag_view = Dataset.open(SOURCE_PATH).get_layer("color").get_mag("1")

    for data_slice_index in range(bbox.size.z):
        for y_tile_index in range(ceil(bbox.size.y / tile_bbox.size.y)):
            for x_tile_index in range(ceil(tile_bbox.size.x / tile_bbox.size.x)):
                tiff_path = (
                    destination_path
                    / f"{data_slice_index + 1}"
                    / f"{y_tile_index + 1}"
                    / f"{x_tile_index + 1}.tiff"
                )

                assert (
                    tiff_path.is_file()
                ), f"Expected a tiff to be written at: {tiff_path}."

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
