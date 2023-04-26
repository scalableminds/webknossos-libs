from pathlib import Path

from webknossos import Dataset

INPUT_FILES = (
    Path("path/to/file1.jpg"),
    Path("path/to/file2.jpg"),
)
OUTPUT_FOLDER = Path("path/to/output_folder")


def main() -> None:
    """Convert a list of images to a wkw dataset."""
    ds = Dataset(dataset_path=OUTPUT_FOLDER, voxel_size=(11, 11, 11), exist_ok=False)
    ds.add_layer_from_images(
        images=INPUT_FILES,
        layer_name="test",
        category="color",
    )


if __name__ == "__main__":
    main()
