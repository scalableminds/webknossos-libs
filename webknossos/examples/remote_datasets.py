from textwrap import wrap

import webknossos as wk


def main() -> None:
    # Remote datasets are read-only, but can be used similar to normal datasets:
    l4_sample_dataset = wk.RemoteDataset.open("l4_sample")

    # Print information of the public l4dense_motta_et_al_demo dataset:
    print(l4_sample_dataset.url)
    print("\n  ".join(["Description:"] + wrap(l4_sample_dataset.description or "")))
    print("Layers:", ", ".join(l4_sample_dataset.layers))
    print("Tags:", ", ".join(l4_sample_dataset.tags))

    # List all accessible remote datasets via get_remote_datasets():
    own_remote_datasets = wk.Dataset.get_remote_datasets()

    # Print the first 10 dataset names from your organization:
    print()
    print(f"First 10 datasets (of {len(own_remote_datasets)}) for own organization:")
    for dataset_name in sorted(own_remote_datasets)[:10]:
        print("*", dataset_name)

    # Set demo tag of l4_sample dataset:
    l4_sample_dataset.tags = ("demo",)

    # List all accessible demo datasets:
    remote_demo_datasets = wk.Dataset.get_remote_datasets(tags="demo")
    print("Remote demo datasets:", list(remote_demo_datasets))
    assert l4_sample_dataset in remote_demo_datasets.values()


if __name__ == "__main__":
    main()
