import numpy as np

import wklib as wk

# A merger mode nml with every tree corresponding to a new merged segment is available.
# All segments in which a node is placed should be merged and saved as a new dataset.

# for local nml:
nml = wk.open("merger-mode.nml")
# wk.open_nml or wk.open_skeleton works, too (and is type-safe)

# for online annotation:
annotation = wk.open_annotation(
    "https://webknossos.brain.mpg.de/annotations/Explorational/60c1dcea01000070000b30d1#7801,8588,1694,0,0.185,6354"
)
nml = annotation.skeleton
# should this save anything to disk, or just happen in memory?

dataset = wk.download(nml.dataset_name, organization=nml.dataset_organization)
# asks for auth token, persisted into .env or similar config file (maybe use xdg-path?)

# sub-part access via dicts or dict-like classes
view = dataset.layers["segmentation"].mag["1"]
# or via getters
view = dataset.get_layer("segmentation").get_mag(1)

segmentation_data = view.read()

for tree in nml.trees():  # nml.trees() is a flattened iterator of all trees
    segment_ids_in_tree = set(
        segmentation_data[tuple(node.position - view.topleft)] for node in tree.nodes
    )
    segment_id_mask = np.isin(segmentation_data, segment_ids_in_tree)
    segmentation_data[segment_id_mask] = min(segment_ids_in_tree)

new_dataset = dataset.copy("new_dataset_path")
# same options as above for reading:
new_dataset.layers["segmentation"].mag[1].write(segmentation_data)
# or
new_dataset.get_layer("segmentation").get_mag("1").write(segmentation_data)
# the data should be downsampled automagically, no need for
# new_dataset.layers["segmentation"].downsample()
