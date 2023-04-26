from pathlib import Path

from webknossos import Dataset

INPUT_FOLDER = Path("path/to/image_files")
OUTPUT_FOLDER = Path("path/to/webknossos_organization")


def main() -> None:
    """Convert a folder of image files to a wkw dataset."""
    Dataset.from_images(
        input_path=INPUT_FOLDER,
        output_path=OUTPUT_FOLDER,
        voxel_size=(11, 11, 11),
        z_slices_sort_key=lambda image_file: image_file.name,
        layer_category="color",
        compress=True,
    )


if __name__ == "__main__":
    main()
