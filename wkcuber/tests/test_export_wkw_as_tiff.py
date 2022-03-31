import os
from math import ceil
from pathlib import Path

import numpy as np
import wkw
from PIL import Image
from webknossos import BoundingBox, Mag
from wkcuber.export_wkw_as_tiff import run, wkw_name_and_bbox_to_tiff_name

DS_NAME = "simple_wk_dataset"
TESTOUTPUT_DIR = Path("testoutput")
SOURCE_PATH = Path("testdata") / DS_NAME


def test_export_tiff_stack() -> None:
    destination_path = os.path.join("testoutput", DS_NAME + "_tiff")
    bbox = BoundingBox((100, 100, 10), (100, 500, 50))
    bbox_dict = bbox.to_config_dict()
    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        destination_path,
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

    test_wkw_file_path = SOURCE_PATH / "color" / Mag(1).to_layer_name()
    with wkw.Dataset.open(str(test_wkw_file_path)) as dataset:
        slice_bbox = bbox_dict
        slice_bbox["size"] = [slice_bbox["size"][0], slice_bbox["size"][1], 1]
        for data_slice_index in range(1, bbox_dict["size"][2] + 1):
            slice_bbox["topleft"] = [
                slice_bbox["topleft"][0],
                slice_bbox["topleft"][1],
                bbox_dict["topleft"][2] + data_slice_index,
            ]
            tiff_path = os.path.join(
                destination_path,
                wkw_name_and_bbox_to_tiff_name("test_export", data_slice_index),
            )

            assert os.path.isfile(
                tiff_path
            ), f"Expected a tiff to be written at: {tiff_path}."

            test_image = np.array(Image.open(tiff_path)).T

            correct_image = dataset.read(
                off=slice_bbox["topleft"], shape=slice_bbox["size"]
            )
            correct_image = np.squeeze(correct_image)

            assert np.array_equal(correct_image, test_image), (
                f"The tiff file {tiff_path} that was written is not "
                f"equal to the original wkw_file."
            )


def test_export_tiff_stack_tile_size() -> None:
    destination_path = os.path.join("testoutput", DS_NAME + "_tile_size")
    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        destination_path,
        "--layer_name",
        "color",
        "--name",
        "test_export",
        "--bbox",
        "0,0,0,100,100,5",
        "--mag",
        "1",
        "--tile_size",
        "30,30",
    ]

    bbox = {"topleft": [0, 0, 0], "size": [100, 100, 5]}

    run(args_list)

    tile_bbox = {"topleft": bbox["topleft"], "size": [30, 30, 1]}
    test_wkw_file_path = SOURCE_PATH / "color" / Mag(1).to_layer_name()
    with wkw.Dataset.open(str(test_wkw_file_path)) as dataset:
        slice_bbox = {"topleft": bbox["topleft"], "size": bbox["size"]}
        slice_bbox["size"] = [slice_bbox["size"][0], slice_bbox["size"][1], 1]
        for data_slice_index in range(bbox["size"][2]):

            for y_tile_index in range(ceil(bbox["size"][1] / tile_bbox["size"][1])):
                for x_tile_index in range(ceil(bbox["size"][0] / tile_bbox["size"][0])):
                    tiff_path = os.path.join(
                        destination_path,
                        f"{data_slice_index + 1}",
                        f"{y_tile_index + 1}",
                        f"{x_tile_index + 1}.tiff",
                    )

                    assert os.path.isfile(
                        tiff_path
                    ), f"Expected a tiff to be written at: {tiff_path}."

                    test_image = np.array(Image.open(tiff_path)).T

                    correct_image = dataset.read(
                        off=[
                            tile_bbox["topleft"][0]
                            + tile_bbox["size"][0] * x_tile_index,
                            tile_bbox["topleft"][1]
                            + tile_bbox["size"][1] * y_tile_index,
                            tile_bbox["topleft"][2] + data_slice_index,
                        ],
                        shape=tile_bbox["size"],
                    )

                    correct_image = np.squeeze(correct_image)

                    assert np.array_equal(correct_image, test_image), (
                        f"The tiff file {tiff_path} that was written "
                        f"is not equal to the original wkw_file."
                    )


def test_export_tiff_stack_tiles_per_dimension() -> None:
    destination_path = os.path.join("testoutput", DS_NAME + "_tiles_per_dimension")
    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        destination_path,
        "--layer_name",
        "color",
        "--name",
        "test_export",
        "--bbox",
        "0,0,0,100,100,5",
        "--mag",
        "1",
        "--tiles_per_dimension",
        "6,6",
    ]

    bbox = {"topleft": [0, 0, 0], "size": [100, 100, 5]}

    run(args_list)

    tile_bbox = {"topleft": bbox["topleft"], "size": [17, 17, 1]}
    test_wkw_file_path = SOURCE_PATH / "color" / Mag(1).to_layer_name()
    with wkw.Dataset.open(str(test_wkw_file_path)) as dataset:
        slice_bbox = bbox
        slice_bbox["size"] = [slice_bbox["size"][0], slice_bbox["size"][1], 1]
        for data_slice_index in range(bbox["size"][2]):

            for y_tile_index in range(ceil(bbox["size"][1] / tile_bbox["size"][1])):
                for x_tile_index in range(
                    ceil(tile_bbox["size"][0] / tile_bbox["size"][0])
                ):
                    tiff_path = os.path.join(
                        destination_path,
                        f"{data_slice_index + 1}",
                        f"{y_tile_index + 1}",
                        f"{x_tile_index + 1}.tiff",
                    )

                    assert os.path.isfile(
                        tiff_path
                    ), f"Expected a tiff to be written at: {tiff_path}."

                    test_image = np.array(Image.open(tiff_path)).T

                    correct_image = dataset.read(
                        off=[
                            tile_bbox["topleft"][0]
                            + tile_bbox["size"][0] * x_tile_index,
                            tile_bbox["topleft"][1]
                            + tile_bbox["size"][1] * y_tile_index,
                            tile_bbox["topleft"][2] + data_slice_index,
                        ],
                        shape=tile_bbox["size"],
                    )

                    correct_image = np.squeeze(correct_image)

                    assert np.array_equal(correct_image, test_image), (
                        f"The tiff file {tiff_path} that was written "
                        f"is not equal to the original wkw_file."
                    )


if __name__ == "__main__":
    test_export_tiff_stack()
    test_export_tiff_stack_tile_size()
    test_export_tiff_stack_tiles_per_dimension()
