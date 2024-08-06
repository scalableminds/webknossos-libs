import webknossos as wk
from webknossos.client.api_client.models import ApiMetadata


def main() -> None:
    with wk.webknossos_context(
        url="https://webknossos.org/",
    ):
        l4_sample_dataset = wk.Dataset.open_remote("l4_sample")
        # Access the metadata of the dataset
        metadata = l4_sample_dataset.metadata
        print(metadata)

        # Edit the metadata of the dataset
        assert metadata is not None
        metadata.append(ApiMetadata(key="new_key", type="string", value="new_value"))
        l4_sample_dataset.metadata = metadata

        # Access metadata of a folder
        folder = l4_sample_dataset.folder
        folder_metadata = folder.metadata
        print(folder_metadata)

        # Edit the metadata of the folder
        folder.set_metadata(metadata=folder_metadata)


if __name__ == "__main__":
    main()
