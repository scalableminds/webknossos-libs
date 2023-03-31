import webknossos as wk


def main() -> None:
    skeleton_a = wk.Skeleton.load("./a.nml")
    skeleton_b = wk.Skeleton.load("./b.nml")

    for tree in skeleton_a.flattened_trees():
        skeleton_b.add_tree(tree)

    skeleton_b.save("./c.nml")


if __name__ == "__main__":
    main()
