from webknossos import DataFormat, Dataset


def main() -> None:
    dataset = Dataset.open_remote(
        "https://webknossos.org/datasets/scalable_minds/cremi_example/"
    )
    print(dataset.layers)
    dataset2 = dataset.copy_dataset("../../cremi3/", voxel_size=dataset.voxel_size)


if __name__ == "__main__":
    main()
