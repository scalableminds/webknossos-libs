# Upload DICOM stack

To convert a DICOM image stack to a WEBKNOSSOS dataset and upload it to [webknossos.org](https://webknossos.org) the bioformats reader is necessary. This example shows how to [create a new dataset from a dicom sequence](../../api/webknossos/dataset/dataset.md#Dataset.from_images), [compress](../../api/webknossos/dataset/dataset.md#Dataset.compress) it and [upload](../../api/webknossos/dataset/dataset.md#Dataset.upload) it to WEBKNOSSOS.

```python
--8<--
webknossos/examples/upload_dicom_stack.py
--8<--
```
