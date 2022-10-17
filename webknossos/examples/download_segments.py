import numpy as np
from tifffile import imwrite

import webknossos as wk

# pylint: disable=unused-variable

ANNOTATION_ID = "634d6e3e010000e000cffad6"
SEGMENT_IDS = [32, 667325]
MAG = wk.Mag("4-4-1")


def main() -> None:
    dataset = wk.Annotation.open_as_remote_dataset(
        ANNOTATION_ID, webknossos_url="https://webknossos.org"
    )
    mag = dataset.get_segmentation_layers()[0].get_mag(MAG)

    z = mag.layer.bounding_box.topleft.z
    with mag.get_buffered_slice_reader() as reader:
        for slice_data in reader:
            slice_data = slice_data[0]  # First channel only
            for segment_id in SEGMENT_IDS:
                segment_mask = (slice_data == segment_id).astype(
                    np.uint8
                ) * 255  # Make a binary mask 0=empty, 255=segment
                segment_mask = segment_mask.T  # Tiff likes the data transposed
                imwrite(
                    f"l4_sample_segments/seg{segment_id:04d}_mag{mag.mag}_z{z:04d}.tiff",
                    segment_mask,
                )
            if z % 100 == 0:
                print(f"z={z:04d}")
            z += 1


if __name__ == "__main__":
    main()
