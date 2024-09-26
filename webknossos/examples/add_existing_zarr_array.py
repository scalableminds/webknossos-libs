from pathlib import Path

import webknossos as wk

OUTPUT_PATH = Path(__file__).parent.parent / "testoutput"
ARRAY_PATH = (
    Path(__file__).parent.parent / "testdata" / "simple_zarr3_dataset" / "color" / "1"
)


def main() -> None:
    ds = wk.Dataset(OUTPUT_PATH, voxel_size=(10, 10, 10))
    layer = ds.add_layer("color", category="color", data_format="zarr3")

    layer.add_mag_from_zarrarray(1, ARRAY_PATH)


if __name__ == "__main__":
    main()
