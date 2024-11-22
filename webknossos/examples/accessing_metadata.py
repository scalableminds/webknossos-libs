import webknossos as wk


def main() -> None:
    with wk.webknossos_context(url="https://webknossos.org/"):
        l4_sample_dataset = wk.Dataset.open_remote("l4_sample")
        # Access the metadata of the dataset
        print(l4_sample_dataset.metadata)

        # Edit the metadata of the dataset
        l4_sample_dataset.metadata["new_key"] = "new_value"

        # Access metadata of a folder
        print(l4_sample_dataset.folder.metadata)

        # Edit the metadata of the folder
        l4_sample_dataset.folder.metadata["new_folder_key"] = "new_folder_value"


if __name__ == "__main__":
    main()
