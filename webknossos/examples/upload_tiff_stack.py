from pathlib import Path

import webknossos as wk


def main() -> None:
    dataset = wk.Dataset.from_images(
        str(Path(__file__).parent.parent / "testdata" / "tiff"),
        "tiff_dataset_upload",
        voxel_size=(12, 12, 12),
    )
    dataset.compress()
    dataset.downsample()

    remote_dataset = dataset.upload()
    url = remote_dataset.url
    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
