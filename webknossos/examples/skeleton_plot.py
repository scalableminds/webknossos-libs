# /// script
# dependencies = [
#   "webknossos[all]",
#   "matplotlib",
# ]
# requires-python = "~=3.12"
# ///

import matplotlib.pyplot as plt
import numpy as np

import webknossos as wk

WK_TOKEN = "..."
ANNOTATION_URL = "https://webknossos.org/annotations/686d28a5010000dd0dd7622f#95564,140740,3232,0,1.949,7662"

with wk.webknossos_context(token=WK_TOKEN):
    a = wk.Annotation.download(ANNOTATION_URL)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")


for tree in a.skeleton.flattened_trees():
    if len(tree.nodes) == 0:
        continue
    # Draw edges
    for u, v in tree.edges:
        edges = np.stack(
            (u.position.to_np() * a.voxel_size, v.position.to_np() * a.voxel_size)
        ).T
        assert tree.color is not None
        ax.plot(*edges, color=tree.color[0:3], alpha=tree.color[3], linewidth=2)

    # # Draw nodes
    # nodes = zip(*[n.position.to_np() * a.voxel_size for n in tree.nodes])
    # ax.scatter(*nodes, s=80, color=tree.color[0:3], alpha=tree.color[3], depthshade=True)

# Set X and Y labels
ax.set_xlabel("X [nm]")
ax.set_ylabel("Y [nm]")
ax.grid(False)

# Top down, rotated by 180deg, flipped Y axis
ax.set_proj_type("ortho")
ax.view_init(elev=90, azim=180)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)

# Hide Z axis
ax.zaxis.set_visible(False)
ax.set_zticks([])
ax.set_zticklabels([])

# plt.show()
plt.savefig("plot.png", dpi=300, bbox_inches="tight", transparent=True)
