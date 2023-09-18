from pathlib import Path

from tifffile import imwrite

import webknossos as wk

# Specify a bounding box for cutouts
# (topleft_x, topleft_y, topleft_z), (width, height, depth)
BOUNDING_BOX = wk.BoundingBox((0, 0, 0), (500, 500, 50))


def load_annotation():
    # Read the WEBKNOSSOS annotation file (a zipped WKW)
    path_to_annotation_file = Path("my_annotation_file.zip")
    annotation = wk.Annotation.load(path_to_annotation_file)

    # Treat it as a regular WK volume layer
    with annotation.temporary_volume_layer_copy() as segmentation_layer:
        # Do any standard layer operation, e.g. reading a cutout as a numpy array
        mag_view = segmentation_layer.get_finest_mag()
        segments = mag_view.read(absolute_bounding_box=BOUNDING_BOX)

        # Write segmentation IDs to an OME Tiff file
        imwrite(
            "segmentation.ome.tiff",
            segments.T,  # note, the tiff lib use different channel order
            ome=True,
            metadata={
                "axes": "ZYXC",
            },
        )


if __name__ == "__main__":
    load_annotation()
