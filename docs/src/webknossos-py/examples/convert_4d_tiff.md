# Convert 4D Tiff

This example demonstrates the basic interactions with Datasets that have more than three dimensions. 

In order to manipulate 4D data in WEBKNOSSOS, we first convert the 4D Tiff dataset into a Zarr3 dataset. This conversion is achieved using the [from_images method](../../api/webknossos/dataset/dataset.md#webknossos.dataset.Dataset.from_images).

Once the dataset is converted, we can access specific layers and views, [read data](../../api/webknossos/dataset/layer/view/mag_view.md#webknossos.dataset.MagView.read) from a defined bounding box, and [write data](../../api/webknossos/dataset/layer/view/mag_view.md#webknossos.dataset.MagView.write) to a different position within the dataset. The [NDBoundingBox](../../api/webknossos/geometry/nd_bounding_box.md#webknossos.geometry.NDBoundingBox) is utilized to select a 4D region of the dataset.

```python
--8<--
webknossos/examples/convert_4d_tiff.py
--8<--
```
