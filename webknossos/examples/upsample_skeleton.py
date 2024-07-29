from pathlib import Path

import webknossos as wk

INPUT_PATH = Path(__file__).parent.parent / "testdata" / "nmls" / "test_a.nml"
OUTPUT_PATH = Path(__file__).parent.parent / "testoutput" / "nmls" / "test_a_scaled.nml"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FACTOR = wk.Vec3Int(4, 4, 1)


def main() -> None:
    """
    Scales a skeleton by a given factor. The factor is applied to the voxel size and the node positions.

    This can be usefull when the skeletons are created in a downsampled version and need to be scaled back to the finest mag.
    """
    skeleton = wk.Skeleton.load(INPUT_PATH)

    skeleton.voxel_size = (
        FACTOR[0] * skeleton.voxel_size[0],
        FACTOR[1] * skeleton.voxel_size[1],
        FACTOR[2] * skeleton.voxel_size[2],
    )
    for tree in skeleton.flattened_trees():
        for node in tree.nodes:
            old_pos = node.position
            node.position = wk.Vec3Int(
                old_pos[0] * FACTOR[0], old_pos[1] * FACTOR[1], old_pos[2] * FACTOR[2]
            )

    skeleton.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
