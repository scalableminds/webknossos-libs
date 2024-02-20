# Convert 4D Tiff

This example uses the [from_images method](../../api/webknossos/dataset/dataset.md#Dataset.from_images) to creates a Zarr3 dataset from a 4D TIFF image, accesses specific layers and views, [reads data](../../api/webknossos/dataset/mag_view.md#MagView.read) within a bounding box, and [writes data](../../api/webknossos/dataset/mag_view.md#MagView.write) to a different position within the dataset. 


```python
--8<--
webknossos/examples/convert_4d_tiff.py
--8<--
```
