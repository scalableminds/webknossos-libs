import webknossos as wk


def main() -> None:
    with wk.webknossos_context(
        url="https://webknossos.org/",
    ):
        l4_sample_dataset = wk.Dataset.open_remote("l4_sample")
        # Access the metadata of the dataset
        dataset_metadata = l4_sample_dataset.metadata
        print(dataset_metadata)

        # Edit the metadata of the dataset
        dataset_metadata["new_key"] = "new_value"
        l4_sample_dataset.metadata = dataset_metadata

        # Access metadata of a folder
        folder_metadata = l4_sample_dataset.folder.metadata
        print(folder_metadata)

        # Edit the metadata of the folder
        folder_metadata["new_folder_key"] = "new_folder_value"
        l4_sample_dataset.folder.metadata = folder_metadata


if __name__ == "__main__":
    main()
