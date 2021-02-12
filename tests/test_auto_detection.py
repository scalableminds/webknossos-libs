from argparse import Namespace

from wkcuber.converter import ImageStackConverter, KnossosConverter

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


def test_knossos_dataset_name_and_layer_path_detection() -> None:
    converter = KnossosConverter()

    # test if dataset name and layer name and mag are correct
    converter.source_files = [
        "knossos/color/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
        "knossos/color/1/x0000/y0000/z0001/test_mag1_x0000_y0000_z0001.raw",
        "knossos/color/1/x0000/y0001/z0000/test_mag1_x0000_y0001_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "knossos"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == "knossos/color"
    assert list(layer_paths.values())[0] == "1"

    # test if in subfolder
    converter.source_files = [
        "superfolder/superfolder/knossos/color/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "knossos"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == "superfolder/superfolder/knossos/color"
    assert list(layer_paths.values())[0] == "1"

    # test for multiple layer
    converter.source_files = [
        "knossos/color/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
        "knossos/segmentation/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "knossos"
    assert len(layer_paths) == 2
    assert "knossos/color" in layer_paths.keys()
    assert "knossos/segmentation" in layer_paths.keys()
    assert all(map(lambda m: m == "1", layer_paths.values()))

    # test if only layer folder given
    converter.source_files = [
        "color/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "dataset"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == "color"
    assert list(layer_paths.values())[0] == "1"

    # test if only mag folder given
    converter.source_files = [
        "1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "dataset"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == ""
    assert list(layer_paths.values())[0] == "1"

    # test if already in mag folder
    converter.source_files = [
        "x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "dataset"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == ""
    assert list(layer_paths.values())[0] == ""

    # test for absolute path
    converter.source_files = [
        "/knossos/color/1/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "knossos"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == "/knossos/color"
    assert list(layer_paths.values())[0] == "1"

    # test if too short absolute path gets detected
    converter.source_files = [
        "/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    assertion_error = False
    try:
        _, _ = converter.detect_dataset_and_layer_paths_with_mag()
    except AssertionError:
        assertion_error = True
    assert assertion_error

    # test for multiple mags
    converter.source_files = [
        "knossos/color/2/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
        "knossos/color/4/x0000/y0000/z0000/test_mag1_x0000_y0000_z0000.raw",
    ]
    dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
    assert dataset_name == "knossos"
    assert len(layer_paths) == 1
    assert list(layer_paths.keys())[0] == "knossos/color"
    assert list(layer_paths.values())[0] == "2"
