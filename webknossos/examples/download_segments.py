import numpy as np
from tifffile import imwrite

import webknossos as wk

ANNOTATION_ID = "634e8fe1010000b4006f3cf4"
SEGMENTATION_LAYER_NAME = "segmentation"
SEGMENT_IDS = [32, 667325]
MAG = wk.Mag("8-8-2")


def main() -> None:
    dataset = wk.Annotation.open_as_remote_dataset(
        ANNOTATION_ID, webknossos_url="https://webknossos.org"
    )
    mag_view = dataset.get_segmentation_layer(SEGMENTATION_LAYER_NAME).get_mag(MAG)

    z = mag_view.bounding_box.topleft.z
    with mag_view.get_buffered_slice_reader() as reader:
        for slice_data in reader:
            slice_data = slice_data[0]  # First channel only
            for segment_id in SEGMENT_IDS:
                segment_mask = (slice_data == segment_id).astype(
                    np.uint8
                ) * 255  # Make a binary mask 0=empty, 255=segment
                segment_mask = segment_mask.T  # Tiff likes the data transposed
                imwrite(
                    f"l4_sample_segments/seg{segment_id:04d}_mag{MAG}_z{z:04d}.tiff",
                    segment_mask,
                )

            print(f"Downloaded z={z:04d}")
            z += MAG.z


if __name__ == "__main__":
    main()
