from textwrap import wrap

import webknossos as wk


def main() -> None:
    # Remote datasets are read-only, but can be used similar to normal datasets:
    l4_sample_dataset = wk.Dataset.open_remote(
        "https://webknossos.org/datasets/scalable_minds/l4dense_motta_et_al_demo"
    )

    # Print information of the public l4dense_motta_et_al_demo dataset:
    print(l4_sample_dataset.url)
    print("\n  ".join(["Description:"] + wrap(l4_sample_dataset.description or "")))
    print("Layers:", ", ".join(l4_sample_dataset.layers))
    print("Tags:", ", ".join(l4_sample_dataset.tags))

    # List all accessible remote datasets via get_remote_dataset_registry():
    dataset_registry = wk.Dataset.get_remote_dataset_registry()
    current_organization = wk.User.get_current_user().organization_id

    # Print the first 10 dataset names from your organization:
    print()
    print("First 10 datasets for own organization:")
    for dataset_name in sorted(dataset_registry[current_organization])[:10]:
        print("*", dataset_name)

    assert (
        l4_sample_dataset
        == dataset_registry["scalable_minds"]["l4dense_motta_et_al_demo"]
    )
    assert l4_sample_dataset in dataset_registry["scalable_minds"].by_tag["demo"]
    assert (
        l4_sample_dataset
        == dataset_registry["scalable_minds"].by_display_name["L4 Mouse Cortex Demo"]
    )


if __name__ == "__main__":
    main()
