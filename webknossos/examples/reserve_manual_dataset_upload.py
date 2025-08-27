import webknossos as wk
from webknossos.client.api_client.models import (
    ApiBoundingBox,
    ApiDataLayerWithPaths,
    ApiDataSourceWithPaths,
    ApiLayerAttachment,
    ApiLayerAttachments,
    ApiMagLocator,
    ApiVoxelSize,
)
from webknossos.dataset.remote_folder import RemoteFolder


def main() -> None:
    # test_dataset()
    # test_attachment()
    test_finish_attachment()


def test_dataset():
    with wk.webknossos_context("http://localhost:9000", token="secretSampleUserToken"):
        # Get the folder id of the upload destination:
        folder_id = RemoteFolder.get_by_path("Datasets/").id

        # Reserve the manual upload of a new dataset:
        dataset_id, data_source = wk.Dataset.reserve_manual_upload(
            dataset_name="my_new_dataset_name3",
            data_source=ApiDataSourceWithPaths(
                data_layers=[
                    ApiDataLayerWithPaths(
                        name="color",
                        category="color",
                        dataFormat="zarr3",
                        element_class="uint8",
                        bounding_box=ApiBoundingBox((0, 0, 0), 10, 10, 10),
                        mags=[ApiMagLocator(mag=(1, 1, 1), path="./1")],
                        attachments=ApiLayerAttachments(
                            agglomerates=[
                                ApiLayerAttachment(
                                    name="agglomerate_view_9000",
                                    dataFormat="zarr3",
                                    path="./agglomerate_view_9000",
                                )
                            ]
                        ),
                    )
                ],
                scale=ApiVoxelSize(unit="nm", factor=(5, 5, 5)),
                status="",
            ),
            initial_team_ids=[],
            folder_id=folder_id,
        )

        print(data_source)

        # Move/Copy the dataset manually to the datastore

        # Mark manual upload as finished
        # print("marking as finished...")
        # wk.Dataset.finish_manual_upload(dataset_id)

        # Dataset should now be available.


def test_attachment():

    with wk.webknossos_context("http://localhost:9000", token="secretSampleUserToken"):
        from webknossos.client.context import _get_context

        context = _get_context()
        path_str = context.api_client_with_auth.dataset_reserve_manual_attachment_upload(
            "68aebb09d50000c9255670ff",
            "segmentation",
            "agglomerate_view_9001",
            "agglomerate",
            "zarr3"
        )
        print(path_str)


def test_finish_attachment():
    with wk.webknossos_context("http://localhost:9000", token="secretSampleUserToken"):
        from webknossos.client.context import _get_context

        context = _get_context()
        context.api_client_with_auth.dataset_finish_manual_attachment_upload(
            "68aebb09d50000c9255670ff",
            "segmentation",
            "agglomerate_view_9001",
            "agglomerate",
            "zarr3"
        )
        print("Finished!")



if __name__ == "__main__":
    main()
