from tifffile import imwrite

import webknossos as wk

DATASET_NAME = "l4_sample"
LAYER_NAME = "color"
MAG = wk.Mag("16-16-4")


def main() -> None:
    dataset = wk.RemoteDataset.open(
        DATASET_NAME,
        organization_id="scalable_minds",
        webknossos_url="https://webknossos.org",
    )
    mag_view = dataset.get_layer(LAYER_NAME).get_mag(MAG)

    z = mag_view.bounding_box.topleft.z
    with mag_view.get_buffered_slice_reader() as reader:
        for slice_data in reader:
            slice_data = slice_data[0]  # First channel only
            slice_data = slice_data.T  # Tiff likes the data transposed

            imwrite(
                f"l4_sample_tiff/mag{MAG}_z{z:04d}.tiff",
                slice_data,
            )

            print(f"Downloaded z={z:04d}")
            z += MAG.z


if __name__ == "__main__":
    main()
