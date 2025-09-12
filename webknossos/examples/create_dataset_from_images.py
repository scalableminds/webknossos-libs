from pathlib import Path

from webknossos import COLOR_CATEGORY, Dataset

INPUT_DIR = Path(__file__).parent.parent / "testdata" / "tiff"
OUTPUT_DIR = Path("testoutput/my_tiff_dataset")


def main() -> None:
    """Convert a folder of image files to a WEBKNOSSOS dataset."""
    dataset = Dataset.from_images(
        input_path=INPUT_DIR,
        output_path=OUTPUT_DIR,
        voxel_size=(11, 11, 11),
        layer_category=COLOR_CATEGORY,
        compress=True,
    )

    print(f"Saved {dataset.name} at {dataset.path}.")

    # dataset.upload()


if __name__ == "__main__":
    main()
