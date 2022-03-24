import webknossos as wk


def main() -> None:
    # Set the correct dataset name and organization where the dataset is saved on webknossos
    # You can copy your organization id from https://webknossos.org/auth/token
    name = "l4dense_motta_et_al_demo"
    organization_id = "scalable_minds"

    # Choose layers and mags to download
    layers = ["color"]
    mags = [wk.Mag(1)]

    # Set the bounding box which should be downloaded
    bbox = wk.BoundingBox([50, 50, 50], [50, 50, 50])

    # Download the datset
    ds = wk.Dataset.download(
        dataset_name_or_url=name,
        organization_id=organization_id,
        bbox=bbox,
        layers=layers,
        mags=mags,
        webknossos_url="https://webknossos.org",
    )

    # Check that the layer was downloaded
    print(f"Layers: {ds.layers}")


if __name__ == "__main__":
    main()
