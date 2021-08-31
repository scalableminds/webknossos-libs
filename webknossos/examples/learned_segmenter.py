from skimage.future import TrainableSegmenter

import webknossos as wk

# should this be possible in memory? download should work, too
annotation = wk.open_annotation(
    "https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640"
)
volume_annotation = (
    annotation.save_volume_annotation_and_get_layer()
)  # returns the layer of the extracted dataset on disk

dataset = wk.download(annotation.dataset_name)
# asks for auth token and default organization, persisted into .env or similar config file (maybe use xdg-path?)

mag = min(volume_annotation.mags)
mag_view = dataset.get_color_layer().mags[mag]
img_data = mag_view.read(volume_annotation.bbox)  # the bbox should be mag-aware

segmenter = TrainableSegmenter()
segmenter.fit(img_data, volume_annotation.mags[mag].read())
segmentation = segmenter.predict(img_data)

result_dataset = dataset.copy("result", subset=volume_annotation.bbox)
result_dataset.add_layer("segmentation", data=segmentation, compressed=True)
# Automatically adds a mag with the specified data and downsamples it.
# Also, it is inferred from the name & dtype that this should be a segmentation.

result_dataset.upload("my_new_segmentation")
