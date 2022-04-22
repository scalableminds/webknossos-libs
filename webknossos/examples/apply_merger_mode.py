from os import environ
from typing import Tuple, cast

import fastremap

import webknossos as wk


def main() -> None:

    ####################################
    # Opening a merger mode annotation #
    ####################################

    nml = wk.Annotation.download(
        "https://webknossos.org/annotations/Explorational/6204d2cd010000db0003db91"
    ).skeleton

    ###############################################
    # Download and open the corresponding dataset #
    ###############################################

    dataset = wk.Dataset.download(
        "l4_sample_dev",
        "scalable_minds",
        path="testoutput/l4_sample_dev",
        webknossos_url="https://webknossos.org",
    )
    in_layer = cast(wk.SegmentationLayer, dataset.get_layer("segmentation"))
    in_mag1 = in_layer.get_mag("1")

    ##############################
    # Compute segment id mapping #
    ##############################

    segment_id_mapping = {}
    for tree in nml.flattened_trees():
        base = None
        for node in tree.nodes:
            segment_id = in_mag1.read(
                absolute_offset=node.position, size=(1, 1, 1)
            ).item()
            if base is None:
                base = segment_id
            segment_id_mapping[segment_id] = base

    print(
        f"Found {len(list(nml.flattened_trees()))} segment id groups with {len(segment_id_mapping)} nodes"
    )
    print(segment_id_mapping)

    ############################
    # Creating an output layer #
    ############################

    out_layer = dataset.add_layer(
        "segmentation_remapped",
        wk.SEGMENTATION_CATEGORY,
        dtype_per_layer=in_layer.dtype_per_layer,
        largest_segment_id=in_layer.largest_segment_id,
    )
    out_mag1 = out_layer.add_mag("1", compress=True)
    out_layer.bounding_box = in_layer.bounding_box

    ###################
    # Apply remapping #
    ###################

    def apply_mapping_for_chunk(args: Tuple[wk.View, wk.View, int]) -> None:
        (in_view, out_view, _) = args
        cube_data = in_view.read()[0]
        # pylint: disable=c-extension-no-member
        fastremap.remap(
            cube_data,
            segment_id_mapping,
            preserve_missing_labels=True,
            in_place=True,
        )
        out_view.write(cube_data)

    in_mag1.for_zipped_chunks(apply_mapping_for_chunk, out_mag1)

    ########################################
    # Optionally, downsample and re-upload #
    ########################################

    if "PYTEST_CURRENT_TEST" not in environ:
        out_layer.downsample()
        dataset.delete_layer("segmentation")
        dataset.upload(
            "l4_sample_remapped",
            layers_to_link=[
                wk.LayerToLink(
                    dataset_name="l4_sample_dev",
                    layer_name="color",
                    organization_id="scalable_minds",
                )
            ],
        )


if __name__ == "__main__":
    main()
