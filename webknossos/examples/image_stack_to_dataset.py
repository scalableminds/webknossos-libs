from pathlib import Path

from webknossos import COLOR_CATEGORY, Dataset

INPUT_FILES = (
    Path(__file__).parent.parent / "testdata" / "tiff" / "test.0000.tiff",
    Path(__file__).parent.parent / "testdata" / "tiff" / "test.0001.tiff",
)
OUTPUT_FOLDER = Path("testoutput/tiff_dataset")


def main() -> None:
    """Convert a list of images into a WEBKNOSSOS dataset and directly add them as a new layer."""
    dataset = Dataset(
        dataset_path=OUTPUT_FOLDER,
        voxel_size=(11, 11, 11),
        name="My_new_dataset",
        exist_ok=False,
    )
    dataset.add_layer_from_images(
        images=INPUT_FILES,
        layer_name="test",
        category=COLOR_CATEGORY,
    )

    # dataset.upload()


if __name__ == "__main__":
    main()
