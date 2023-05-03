from pathlib import Path

from webknossos import Dataset
from webknossos.dataset import COLOR_CATEGORY

INPUT_DIR = Path("webknossos/testdata/tiff")
OUTPUT_DIR = Path("../example_output/my_tiff_dataset")


def main() -> None:
    """Convert a folder of image files to a WEBKNOSSOS dataset."""
    dataset = Dataset.from_images(
        input_path=INPUT_DIR,
        output_path=OUTPUT_DIR,
        voxel_size=(11, 11, 11),
        layer_category=COLOR_CATEGORY,
        compress=True,
    )
    dataset.upload()


if __name__ == "__main__":
    main()
