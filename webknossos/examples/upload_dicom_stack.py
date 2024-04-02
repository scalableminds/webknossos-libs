from pathlib import Path
from time import gmtime, strftime

import webknossos as wk


def main() -> None:
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    dataset = wk.Dataset.from_images(
        str(Path(__file__).parent.parent / "testdata" / "dicoms"),
        f"dicom_dataset_{time_str}",
        voxel_size=(12, 12, 12),
    )
    dataset.compress()

    remote_dataset = dataset.upload()
    url = remote_dataset.url
    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
