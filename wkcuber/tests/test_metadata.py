from pathlib import Path

import numpy as np
import wkw
from wkcuber.metadata import (
    detect_mappings,
    read_datasource_properties,
    read_metadata_for_layer,
    refresh_metadata,
    write_webknossos_metadata,
)
from wkcuber._internal.utils import WkwDatasetInfo, ensure_wkw, open_wkw

from .constants import TESTDATA_DIR


def test_element_class_conversion(tmp_path: Path) -> None:
    test_wkw_path = tmp_path / "test_metadata"
    prediction_layer_name = "prediction"
    prediction_wkw_info = WkwDatasetInfo(
        test_wkw_path,
        prediction_layer_name,
        1,
        wkw.Header(np.float32, num_channels=3, file_len=1),
    )
    ensure_wkw(prediction_wkw_info)

    write_custom_layer(test_wkw_path, "prediction", np.float32, num_channels=3)
    write_webknossos_metadata(
        test_wkw_path,
        "test_metadata",
        (11.24, 11.24, 28),
        compute_max_id=True,
        exact_bounding_box={"topLeft": [0, 0, 0], "width": 4, "height": 4, "depth": 4},
    )
    write_custom_layer(test_wkw_path, "segmentation", np.float64, num_channels=1)
    write_custom_layer(test_wkw_path, "color", np.uint8, num_channels=3)

    refresh_metadata(test_wkw_path)

    check_element_class_of_layer(test_wkw_path, "prediction", "float", np.float32)
    check_element_class_of_layer(test_wkw_path, "segmentation", "double", np.float64)
    check_element_class_of_layer(test_wkw_path, "color", "uint24", np.uint8)


def check_element_class_of_layer(
    test_wkw_path: Path,
    layer_name: str,
    expected_element_class: str,
    expected_dtype: type,
) -> None:
    datasource_properties = read_datasource_properties(test_wkw_path)
    layer_to_check = None
    for layer in datasource_properties["dataLayers"]:
        if layer["name"] == layer_name:
            layer_to_check = layer

    assert (
        layer_to_check
    ), f"Did not find layer {layer_name} in datasource_properties.json."
    assert layer_to_check["elementClass"] == expected_element_class
    _, converted_dtype, _, _ = read_metadata_for_layer(test_wkw_path, layer_name)
    assert converted_dtype == expected_dtype


def write_custom_layer(
    target_path: Path, layer_name: str, dtype: type, num_channels: int
) -> None:
    data = (
        np.arange(4 * 4 * 4 * num_channels)
        .reshape((num_channels, 4, 4, 4))
        .astype(dtype)
    )
    prediction_wkw_info = WkwDatasetInfo(
        target_path, layer_name, 1, wkw.Header(dtype, num_channels, file_len=1)
    )
    ensure_wkw(prediction_wkw_info)
    with open_wkw(prediction_wkw_info) as dataset:
        dataset.write(off=(0, 0, 0), data=data)


def test_mapping_detection(WT1_path: Path) -> None:
    # NOTE: the mappings do not match do the actual wkw data. Therefore do not use them
    expected_mappings = [
        "test_mapping_1.json",
        "test_mapping_2.json",
        "test_mapping_3.json",
        "test_mapping_4.json",
        "test_mapping_5.json",
    ]
    datapath_with_mappings = TESTDATA_DIR / "test_metadata"
    layer_name_with_mapping = "segmentation"
    detected_mappings = detect_mappings(datapath_with_mappings, layer_name_with_mapping)

    # test if all detected mappings are int the expected_mappings
    assert all(
        detected_mapping in expected_mappings for detected_mapping in detected_mappings
    ), "Found unexpected mapping(s)."
    # test if all mappings were detected
    assert len(expected_mappings) == len(
        detected_mappings
    ), "Did not find all mappings."

    datapath_without_mappings = WT1_path
    layer_name_without_mapping = "color"
    detected_mappings = detect_mappings(
        datapath_without_mappings, layer_name_without_mapping
    )
    assert (
        len(detected_mappings) == 0
    ), "Found mapping(s) even though there should not be any mapping."
