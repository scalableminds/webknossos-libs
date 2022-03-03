import os
from functools import partial
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
    training_data_bbox = wk.BoundingBox.from_tuple6(
        annotation.user_bounding_boxes[0]  # type: ignore[index]
    )
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    new_dataset_name = annotation.dataset_name + f"_segmented_{time_str}"
    dataset = wk.Dataset.download(
        annotation.dataset_name,
        organization_name="scalable_minds",
        path=new_dataset_name,
        webknossos_url="https://webknossos.org",
    )
    dataset.name = new_dataset_name

    volume_annotation = annotation.export_volume_layer_to_dataset(dataset)
    volume_annotation.bounding_box = training_data_bbox

    mag = wk.Mag(1)
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
    Y_train = np.squeeze(volume_annotation.mags[mag].read())

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
    segmentation_layer = dataset.add_layer(
        "segmentation",
        wk.SEGMENTATION_CATEGORY,
        segmentation.dtype,
        compressed=True,
        largest_segment_id=int(segmentation.max()),
    )
    segmentation_layer.bounding_box = dataset.layers["color"].bounding_box
    segmentation_layer.add_mag(mag, compress=True).write(segmentation)

    url = dataset.upload(
        layers_to_link=[
            wk.LayerToLink(
                organization_name="scalable_minds",
                dataset_name=annotation.dataset_name,
                layer_name="color",
            )
        ]
        if "PYTEST_CURRENT_TEST" not in os.environ
        else None
    )

    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
