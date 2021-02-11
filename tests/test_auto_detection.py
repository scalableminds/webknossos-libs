from argparse import Namespace

from wkcuber.converter import ImageStackConverter

TEST_NAMESPACE = Namespace()
TEST_NAMESPACE.target_path = "a/random/path"


def test_tiff_dataset_name_and_layer_name_detection() -> None:
    converter = ImageStackConverter()

    # test if ds name and layer name are correctly detected
    converter.source_files = [
        "test/color/001.tif",
        "test/color/002.tif",
        "test/color/003.tif",
    ]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "test"
    assert len(layer_names) == 1
    assert layer_names[0] == "color"

    # test if in subfolder
    converter.source_files = [
        "superfolder/test/color/001.tif",
        "superfolder/test/color/002.tif",
        "superfolder/test/color/003.tif",
    ]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "test"
    assert len(layer_names) == 1
    assert layer_names[0] == "color"

    # test for multiple layers
    converter.source_files = [
        "test/color/001.tif",
        "test/color/002.tif",
        "test/color/003.tif",
        "test/segmentation/001.tif",
        "test/segmentation/002.tif",
        "test/segmentation/003.tif",
    ]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "test"
    assert len(layer_names) == 2
    assert "color" in layer_names
    assert "segmentation" in layer_names

    # test if in single folder and folder name is layer name
    converter.source_files = [
        "color/001.tif",
        "color/002.tif",
        "color/003.tif",
    ]
    converter.args = TEST_NAMESPACE
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "path"
    assert len(layer_names) == 1
    assert layer_names[0] == "color"

    # test if in single folder and folder name is ds name
    converter.source_files = [
        "dataset/001.tif",
        "dataset/002.tif",
        "dataset/003.tif",
    ]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "dataset"
    assert len(layer_names) == 1
    assert layer_names[0] == "color"

    # test if single file in folder
    converter.source_files = ["dataset/brain.tif"]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "dataset"
    assert len(layer_names) == 1
    assert layer_names[0] == "brain"

    # test if single file
    converter.source_files = ["brain.tif"]
    dataset_name, layer_names = converter.detect_dataset_and_layer_names()
    assert dataset_name == "brain"
    assert len(layer_names) == 1
    assert layer_names[0] == "color"
