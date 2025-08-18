import webknossos as wk
from webknossos.client.api_client.models import ApiDataSourceWithPaths, ApiBoundingBox, \
    ApiDataLayerWithPaths, ApiMagLocator, ApiLayerAttachment, ApiLayerAttachments, ApiVoxelSize
from webknossos.dataset.remote_folder import RemoteFolder


def main() -> None:
    with wk.webknossos_context("http://localhost:9000", token="secretSampleUserToken"):
        # Get the folder id of the upload destination:
        folder_id = RemoteFolder.get_by_path("Datasets/").id

        # Reserve the manual upload of a new dataset:
        dataset_id, data_source = wk.Dataset.reserve_manual_upload(
            dataset_name="my_new_dataset_name",
            data_source=ApiDataSourceWithPaths(
                data_layers=[
                    ApiDataLayerWithPaths(
                        name="color",
                        category="color",
                        element_class="uint8",
                        bounding_box=ApiBoundingBox((0, 0, 0), 10, 10, 10),
                        mags=[
                            ApiMagLocator(
                                mag=(1, 1, 1),
                                path="./1"
                            )
                        ],
                        attachments=ApiLayerAttachments(
                            agglomerates=[
                                ApiLayerAttachment(
                                    name="agglomerate_view_9000",
                                    dataFormat="zarr3",
                                    path="./agglomerate_view_9000"
                                )
                            ]
                        )
                    )
                ],
                scale=ApiVoxelSize(unit="nm", factor=(5,5,5))
            ),
            initial_team_ids=[],
            folder_id=folder_id
        )

        print(data_source)

        # Move/Copy the dataset manually to the datastore

        # Mark manual upload as finished
        wk.Dataset.finish_manual_upload(dataset_id)

        # Dataset should now be available.


if __name__ == "__main__":
    main()
