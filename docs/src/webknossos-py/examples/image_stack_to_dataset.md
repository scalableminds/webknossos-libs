# Image Stack to Dataset

This example shows how to [create a new WEBKNOSSOS dataset from a stack of images](../../api/webknossos/dataset/dataset.md#webknossos.dataset.Dataset.from_images), e.g. Tiff, JPEG, etc files.

There are a few assumptions we made about the images used for this example:

- all images have the same size
- they have the same dtype (e.g. `uint8` or `float`)
- they are greyscale images from microscopy / MRI / CT scan, therefore the category is `color`
- masks and segmentations are not included yet

```python
--8<--
webknossos/examples/image_stack_to_dataset.py
--8<--
```
