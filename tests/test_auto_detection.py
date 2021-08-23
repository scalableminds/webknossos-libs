from wkcuber.converter import ImageStackConverter, KnossosConverter
from os.path import sep

TEST_PREFIXES = ["", sep, f"..{sep}"]


def test_tiff_dataset_name_and_layer_name_detection() -> None:
    for prefix in TEST_PREFIXES:

        # test if ds name and layer name are correctly detected
        converter = ImageStackConverter()
        converter.source_files = [
            f"{prefix}test{sep}color{sep}001.tif",
            f"{prefix}test{sep}color{sep}002.tif",
            f"{prefix}test{sep}color{sep}003.tif",
        ]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "test"
        assert len(layer_path_to_layer_name) == 1
        assert list(layer_path_to_layer_name)[0] == f"{prefix}test{sep}color"
        assert list(layer_path_to_layer_name.values())[0] == "color"

        # test if in subfolder
        converter = ImageStackConverter()
        converter.source_files = [
            f"{prefix}superfolder{sep}test{sep}color{sep}001.tif",
            f"{prefix}superfolder{sep}test{sep}color{sep}002.tif",
            f"{prefix}superfolder{sep}test{sep}color{sep}003.tif",
        ]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "test"
        assert len(layer_path_to_layer_name) == 1
        assert (
            list(layer_path_to_layer_name)[0]
            == f"{prefix}superfolder{sep}test{sep}color"
        )
        assert list(layer_path_to_layer_name.values())[0] == "color"

        # test for multiple layers
        converter = ImageStackConverter()
        converter.source_files = [
            f"{prefix}test{sep}color{sep}001.tif",
            f"{prefix}test{sep}color{sep}002.tif",
            f"{prefix}test{sep}color{sep}003.tif",
            f"{prefix}test{sep}segmentation{sep}001.tif",
            f"{prefix}test{sep}segmentation{sep}002.tif",
            f"{prefix}test{sep}segmentation{sep}003.tif",
        ]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "test"
        assert len(layer_path_to_layer_name) == 2
        assert f"{prefix}test{sep}color" in layer_path_to_layer_name.keys()
        assert f"{prefix}test{sep}segmentation" in layer_path_to_layer_name.keys()
        assert "color" in layer_path_to_layer_name.values()
        assert "segmentation" in layer_path_to_layer_name.values()

        # test if in single folder and folder name is layer name
        converter = ImageStackConverter()
        converter.source_files = [
            f"{prefix}color{sep}001.tif",
            f"{prefix}color{sep}002.tif",
            f"{prefix}color{sep}003.tif",
        ]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "dataset"
        assert len(layer_path_to_layer_name) == 1
        assert list(layer_path_to_layer_name)[0] == f"{prefix}color"
        assert list(layer_path_to_layer_name.values())[0] == "color"

        # test if in single folder and folder name is ds name
        converter = ImageStackConverter()
        converter.source_files = [
            f"{prefix}test_dataset{sep}001.tif",
            f"{prefix}test_dataset{sep}002.tif",
            f"{prefix}test_dataset{sep}003.tif",
        ]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "test_dataset"
        assert len(layer_path_to_layer_name) == 1
        assert list(layer_path_to_layer_name)[0] == f"{prefix}test_dataset"
        assert list(layer_path_to_layer_name.values())[0] == "color"

        # test if single file in folder
        converter = ImageStackConverter()
        converter.source_files = [f"{prefix}test_dataset{sep}brain.tif"]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "test_dataset"
        assert len(layer_path_to_layer_name) == 1
        assert (
            list(layer_path_to_layer_name)[0] == f"{prefix}test_dataset{sep}brain.tif"
        )
        assert list(layer_path_to_layer_name.values())[0] == "brain"

        # test if single file
        converter = ImageStackConverter()
        converter.source_files = [f"{prefix}brain.tif"]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "brain"
        assert len(layer_path_to_layer_name) == 1
        assert list(layer_path_to_layer_name)[0] == f"{prefix}brain.tif"
        assert list(layer_path_to_layer_name.values())[0] == "color"

        # test for multiple files with no parent directory
        converter = ImageStackConverter()
        converter.source_files = [f"{prefix}001.tif", f"{prefix}002.tif"]
        (
            dataset_name,
            layer_path_to_layer_name,
        ) = converter.detect_dataset_name_and_layer_path_to_layer_name()
        assert dataset_name == "dataset"
        assert len(layer_path_to_layer_name) == 1
        assert list(layer_path_to_layer_name.keys())[0] == prefix
        assert list(layer_path_to_layer_name.values())[0] == "color"


def test_knossos_dataset_name_and_layer_path_detection() -> None:
    for prefix in TEST_PREFIXES:

        # test if dataset name and layer name and mag are correct
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}knossos{sep}color{sep}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
            f"{prefix}knossos{sep}color{sep}1{sep}x0000{sep}y0000{sep}z0001{sep}test_mag1_x0000_y0000_z0001.raw",
            f"{prefix}knossos{sep}color{sep}1{sep}x0000{sep}y0001{sep}z0000{sep}test_mag1_x0000_y0001_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "knossos"
        assert len(layer_paths) == 1
        assert list(layer_paths.keys())[0] == f"{prefix}knossos{sep}color"
        assert list(layer_paths.values())[0] == {"1"}

        # test if in subfolder
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}superfolder{sep}superfolder{sep}knossos{sep}color{sep}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "knossos"
        assert len(layer_paths) == 1
        assert (
            list(layer_paths.keys())[0]
            == f"{prefix}superfolder{sep}superfolder{sep}knossos{sep}color"
        )
        assert list(layer_paths.values())[0] == {"1"}

        # test for multiple layer
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}knossos{sep}color{sep}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
            f"{prefix}knossos{sep}segmentation{sep}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "knossos"
        assert len(layer_paths) == 2
        assert f"{prefix}knossos{sep}color" in layer_paths.keys()
        assert f"{prefix}knossos{sep}segmentation" in layer_paths.keys()
        assert all(map(lambda m: m == {"1"}, layer_paths.values()))

        # test if only layer folder given
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}color{sep}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "dataset"
        assert len(layer_paths) == 1
        assert list(layer_paths.keys())[0] == f"{prefix}color"
        assert list(layer_paths.values())[0] == {"1"}

        # test if only mag folder given
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}1{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "dataset"
        assert len(layer_paths) == 1
        assert list(layer_paths.keys())[0] == prefix
        assert list(layer_paths.values())[0] == {"1"}

        # test if already in mag folder
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "dataset"
        assert len(layer_paths) == 1
        assert list(layer_paths.keys())[0] == prefix
        assert list(layer_paths.values())[0] == {""}

        # test if too short path gets detected
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        assertion_error = False
        try:
            _, _ = converter.detect_dataset_and_layer_paths_with_mag()
        except AssertionError:
            assertion_error = True
        assert assertion_error

        # test for multiple mags
        converter = KnossosConverter()
        converter.source_files = [
            f"{prefix}knossos{sep}color{sep}2{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
            f"{prefix}knossos{sep}color{sep}4{sep}x0000{sep}y0000{sep}z0000{sep}test_mag1_x0000_y0000_z0000.raw",
        ]
        dataset_name, layer_paths = converter.detect_dataset_and_layer_paths_with_mag()
        assert dataset_name == "knossos"
        assert len(layer_paths) == 1
        assert list(layer_paths.keys())[0] == f"{prefix}knossos{sep}color"
        assert list(layer_paths.values())[0] == {"2", "4"}
