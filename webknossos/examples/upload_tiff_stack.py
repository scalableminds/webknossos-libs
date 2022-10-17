from pathlib import Path

import webknossos as wk

# pylint: disable=unused-variable


def main() -> None:
    dataset = wk.Dataset.from_images(
        str(Path(__file__).parent.parent / "testdata" / "tiff_dataset" / "*.tiff"),
        "test_dataset",
        voxel_size=(12, 12, 12),
    )
    dataset.compress()
    dataset.downsample()

    remote_dataset = dataset.upload()
    url = remote_dataset.url
    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
