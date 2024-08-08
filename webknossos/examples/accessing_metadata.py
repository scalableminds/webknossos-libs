import webknossos as wk
from webknossos.client.api_client.models import ApiMetadata


def main() -> None:
    with wk.webknossos_context(
        url="https://webknossos.org/",
    ):
        l4_sample_dataset = wk.Dataset.open_remote("l4_sample")
        # Access the metadata of the dataset
        dataset_metadata = l4_sample_dataset.metadata
        print(dataset_metadata)

        # Edit the metadata of the dataset
        dataset_metadata.append(
            ApiMetadata(key="new_key", type="string", value="new_value")
        )
        l4_sample_dataset.metadata = dataset_metadata

        # Access metadata of a folder
        folder_metadata = l4_sample_dataset.folder.metadata
        print(folder_metadata)

        # Edit the metadata of the folder
        folder_metadata.append(
            ApiMetadata(key="new_folder_key", type="string", value="new_folder_value")
        )
        l4_sample_dataset.folder.metadata = folder_metadata


if __name__ == "__main__":
    main()
