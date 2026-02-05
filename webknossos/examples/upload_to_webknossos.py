import webknossos as wk
from webknossos import LayerToLink
from webknossos.dataset.remote_folder import RemoteFolder


def main() -> None:
    # Open your unpublished dataset
    dataset = wk.Dataset.open("my_dataset.wkw")

    existing_dataset = wk.RemoteDataset.open(
        "https://webknossos.org/datasets/liconn_expid82-678f98e3010000dd017f4449/#3825,4866,795,0,1.3"
    )
    # Publish the dataset to webknossos
    # The data will be uploaded to the upload destination folder
    dataset.upload(
        new_dataset_name="my_new_dataset_name",
        folder=RemoteFolder.get_by_path("Datasets"),
        layers_to_link=[
            LayerToLink.from_remote_layer(
                existing_dataset.get_layer("image_230130b"), new_layer_name="image"
            ),
        ],
    )


if __name__ == "__main__":
    main()
