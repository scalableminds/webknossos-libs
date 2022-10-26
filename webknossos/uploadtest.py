import webknossos as wk

with wk.webknossos_context(url="http://localhost:9000", token="secretSampleUserToken"):
    dataset = wk.Dataset("uploadtest-data", exist_ok=True)

    dataset.upload(
        layers_to_link=[
            wk.LayerToLink(
                organization_id="sample_organization",
                dataset_name="linktest",
                layer_name="images",
            )
        ]
    )
