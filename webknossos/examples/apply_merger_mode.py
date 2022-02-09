from pathlib import Path
from shutil import copyfileobj
from typing import Tuple, cast
from urllib.request import urlopen
from zipfile import ZipFile

import fastremap

import webknossos as wk


def main() -> None:

    ####################################
    # Opening a merger mode annotation #
    ####################################

    nml = wk.Skeleton.load(Path("testdata/annotations/l4_sample__merger_mode.nml"))

    ###############################################
    # Download and open the corresponding dataset #
    ###############################################

    copyfileobj(
        urlopen("https://static.webknossos.org/data/l4_segmentation.zip"),
        open("testdata/l4_segmentation.zip", "wb"),
    )
    ZipFile("testdata/l4_segmentation.zip").extractall("testdata")

    dataset = wk.Dataset.open("testdata/l4_segmentation")
    in_layer = cast(wk.SegmentationLayer, dataset.get_layer("segmentation"))
    in_mag1 = in_layer.get_mag("1")

    ###############################
    # Compute equivalence classes #
    ###############################

    equiv_classes = [
        set(
            in_mag1.read(absolute_offset=node.position, size=(1, 1, 1))[0, 0, 0, 0]
            for node in graph.nodes
        )
        for graph in nml.flattened_graphs()
    ]

    equiv_map = {}
    for segment_ids in equiv_classes:
        base = next(iter(segment_ids))
        for segment_id in segment_ids:
            equiv_map[segment_id] = base

    print(f"Found {len(equiv_classes)} equivalence classes with {len(equiv_map)} nodes")

    ############################
    # Creating an output layer #
    ############################

    if "segmentation_remapped" in dataset.layers:
        dataset.delete_layer("segmentation_remapped")

    out_layer = dataset.add_layer(
        "segmentation_remapped",
        wk.SEGMENTATION_CATEGORY,
        dtype_per_layer=in_layer.dtype_per_layer,
        largest_segment_id=in_layer.largest_segment_id,
    )
    out_mag1 = out_layer.add_mag("1")

    ###################
    # Apply remapping #
    ###################

    def apply_mapping_for_chunk(args: Tuple[wk.View, int]) -> None:
        (view, _) = args
        cube_data = view.read()[0]
        # pylint: disable=c-extension-no-member
        fastremap.remap(
            cube_data,
            equiv_map,
            preserve_missing_labels=True,
            in_place=True,
        )
        out_mag1.write(
            cube_data, absolute_offset=view.bounding_box.in_mag(out_mag1.mag).topleft
        )

    in_mag1.for_each_chunk(apply_mapping_for_chunk)

    ###############################
    # Downsample new segmentation #
    ###############################

    out_layer.downsample()


if __name__ == "__main__":
    main()
