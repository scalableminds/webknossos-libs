from pathlib import Path
import tempfile

import webknossos as wk


def main() -> None:
    print(Path(__file__).parent.parent.parent.parent)
    ds = wk.Dataset.open(
        Path(__file__).parent.parent.parent.parent / "upsample_dataset"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ds2 = ds.copy_dataset(Path(tmpdir) / "upsample_dataset")

        ds2.get_segmentation_layers()[0].upsample(wk.Mag("16-16-8"))


if __name__ == "__main__":
    main()
