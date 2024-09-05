from pathlib import Path

from tifffile import imwrite

import webknossos as wk

# Specify a bounding box for cutouts
# (topleft_x, topleft_y, topleft_z), (width, height, depth)
BOUNDING_BOX = wk.BoundingBox((0, 0, 0), (500, 500, 50))


def load_annotation(annotation_file: Path) -> None:
    # Read the WEBKNOSSOS annotation file (a zipped WKW)
    annotation = wk.Annotation.load(annotation_file)

    # Treat it as a regular WK volume layer
    with annotation.temporary_volume_layer_copy() as segmentation_layer:
        # Do any standard layer operation, e.g. reading a cutout as a numpy array
        mag_view = segmentation_layer.get_finest_mag()
        segments = mag_view.read(absolute_bounding_box=BOUNDING_BOX)

        # Drop channel dimension as it has depth 1
        segments = segments[0]

        # Write segmentation IDs to an OME Tiff file
        imwrite(
            "segmentation.ome.tiff",
            segments.T,  # note, the tiff lib use different channel order
            ome=True,
            metadata={
                "axes": "ZYX",
            },
        )


if __name__ == "__main__":
    # Path to annotation file on your computer
    ANNOTATION_FILE = Path("my_annotation_file.zip")

    load_annotation(ANNOTATION_FILE)
