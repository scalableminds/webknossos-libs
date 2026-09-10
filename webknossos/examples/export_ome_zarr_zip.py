"""
Exports a WEBKNOSSOS dataset layer as a single, self-contained OME-Zarr zip
archive that is compliant with NGFF RFC-9
(https://ngff.openmicroscopy.org/rfc/9/), "Zipped OME-Zarr".

`layer.export.as_ozx` builds the archive from the layer's current mag
pyramid; WEBKNOSSOS already writes OME-NGFF 0.5 multiscale metadata for
every Zarr3 layer (see `webknossos.dataset.ome_metadata`), so no extra
OME-Zarr metadata needs to be written by hand.

`tensorstore`, the Zarr backend used by webknossos-libs, has no zip kvstore
driver, so the resulting archive can't be reopened with `wk.Dataset.open`.
It is meant for other OME-Zarr-compatible tools (e.g. ome-zarr-py, napari,
neuroglancer) instead.
"""

import zipfile
from pathlib import Path

import numpy as np
from upath import UPath

import webknossos as wk

OUTPUT_PATH = Path(__file__).parent.parent / "testoutput" / "rfc9_example_dataset"
ZIP_PATH = Path(__file__).parent.parent / "testoutput" / "color.ozx"


def main() -> None:
    ###############################################
    # Creating a small multiscale WEBKNOSSOS layer #
    ###############################################
    dataset = wk.Dataset(OUTPUT_PATH, voxel_size=(11, 11, 25), exist_ok=True)
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype="uint8",
        data_format="zarr3",
        bounding_box=wk.BoundingBox((0, 0, 0), (256, 256, 256)),
    )
    mag1 = layer.add_mag(1)
    mag1.write(data=(np.random.rand(256, 256, 256) * 255).astype(np.uint8))

    # Generates further mags (2, 4, ...), building a proper multiscale
    # pyramid. WEBKNOSSOS automatically records every mag in the layer's
    # OME-NGFF `multiscales` metadata (see `webknossos.dataset.ome_metadata`).
    layer.downsample()

    ############################################################
    # Packaging the layer as an RFC-9 compliant OME-Zarr zip   #
    ############################################################
    layer.export.as_ozx(output_path=UPath(ZIP_PATH))

    #######################################
    # Sanity-checking the resulting archive #
    #######################################
    with zipfile.ZipFile(ZIP_PATH) as zip_file:
        entries = zip_file.infolist()
        assert entries[0].filename == "zarr.json", (
            "the root zarr.json must be the first zip entry"
        )
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in entries), (
            "all entries must be stored uncompressed"
        )
        print(f"Wrote {len(entries)} entries to {ZIP_PATH}")
        print(f"Archive comment: {zip_file.comment.decode('utf-8')}")


if __name__ == "__main__":
    main()
