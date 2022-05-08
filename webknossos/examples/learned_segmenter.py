import os
from functools import partial
from tempfile import TemporaryDirectory
from time import gmtime, strftime

import numpy as np
from skimage import feature
from skimage.future import TrainableSegmenter

import webknossos as wk

# pylint: disable=unsubscriptable-object


def main() -> None:
    # We are going to use a public demo annotation for this example

    annotation = wk.Annotation.download(
        "https://webknossos.org/annotations/Explorational/616457c2010000870032ced4"
    )

    # Step 1: Download the dataset and our training data annotation from webKnossos to our local computer
    training_data_bbox = annotation.user_bounding_boxes[0]  # type: ignore[index]
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    new_dataset_name = annotation.dataset_name + f"_segmented_{time_str}"
    mag = wk.Mag(1)
    dataset = wk.Dataset.open_remote(
        annotation.dataset_name,
        organization_id="scalable_minds",
        webknossos_url="https://webknossos.org",
    )

    with annotation.temporary_volume_layer_copy() as volume_annotation_layer:
        volume_annotation_data = volume_annotation_layer.mags[mag].read(
            absolute_bounding_box=training_data_bbox
        )

    mag_view = dataset.layers["color"].mags[mag]

    # Step 2: Initialize a machine learning model to segment our dataset
    features_func = partial(
        feature.multiscale_basic_features, multichannel=True, edges=False
    )
    segmenter = TrainableSegmenter(features_func=features_func)

    # Step 3: Manipulate our data to fit the ML model and start training on
    # data from our annotated training data bounding box
    print("Starting training…")
    img_data_train = mag_view.read(
        absolute_bounding_box=training_data_bbox
    )  # wk data has dimensions (Channels, X, Y, Z)
    # move channels to last dimension, remove z dimension to match skimage's shape
    X_train = np.moveaxis(np.squeeze(img_data_train), 0, -1)
    Y_train = np.squeeze(volume_annotation_data)

    segmenter.fit(X_train, Y_train)

    # Step 4: Use our trained model and predict a class for each pixel in the dataset
    # to get a full segmentation of the data
    print("Starting prediction…")
    X_predict = np.moveaxis(np.squeeze(mag_view.read()), 0, -1)
    Y_predicted = segmenter.predict(X_predict)
    segmentation = Y_predicted[:, :, None]  # adds z dimension
    assert segmentation.max() < 256
    segmentation = segmentation.astype("uint8")

    # Step 5: Bundle everying a webKnossos layer and upload to wK for viewing and further work
    with TemporaryDirectory() as tempdir:
        new_dataset = wk.Dataset(
            tempdir, voxel_size=dataset.voxel_size, name=new_dataset_name
        )
        segmentation_layer = new_dataset.add_layer(
            "segmentation",
            wk.SEGMENTATION_CATEGORY,
            segmentation.dtype,
            compressed=True,
            largest_segment_id=int(segmentation.max()),
        )
        segmentation_layer.bounding_box = dataset.layers["color"].bounding_box
        segmentation_layer.add_mag(mag, compress=True).write(segmentation)

        uploaded_dataset = new_dataset.upload(
            layers_to_link=[
                wk.LayerToLink(
                    organization_id="scalable_minds",
                    dataset_name=annotation.dataset_name,
                    layer_name="color",
                )
            ]
            if "PYTEST_CURRENT_TEST" not in os.environ
            else None
        )

    url = uploaded_dataset.url
    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
