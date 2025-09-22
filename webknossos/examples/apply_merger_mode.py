from pathlib import Path

import fastremap

import webknossos as wk

def main() -> None:
    ####################################
    # Opening a merger mode annotation #
    ####################################

    nml = wk.Annotation.download(
        "https://webknossos.org/annotations/6748612b0100001101c81156"
    ).skeleton

    ###############################################
    # Open the corresponding dataset #
    ###############################################

    input_dataset = wk.Dataset.open_remote(
        "l4_sample",
        organization_id="scalable_minds",
    )
    in_layer = input_dataset.get_segmentation_layer("segmentation")
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
    output_dataset = wk.Dataset("testoutput/l4_sample", voxel_size=input_dataset.voxel_size)
    out_layer = output_dataset.add_layer(
        "segmentation_remapped",
        wk.SEGMENTATION_CATEGORY,
        dtype_per_channel=in_layer.dtype_per_channel,
        largest_segment_id=in_layer.largest_segment_id,
    )
    out_mag1 = out_layer.add_mag("1")
    out_layer.bounding_box = in_layer.bounding_box

    ###################
    # Apply remapping #
    ###################

    def apply_mapping_for_chunk(args: tuple[wk.View, wk.View, int]) -> None:
        (in_view, out_view, _) = args
        cube_data = in_view.read()[0]
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

    out_layer.downsample()
    output_dataset.upload(
        "l4_sample_remapped",
        layers_to_link=[
            input_dataset.get_layer("color")
        ],
    )


if __name__ == "__main__":
    main()
