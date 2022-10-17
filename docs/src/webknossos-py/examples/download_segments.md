# Download segment masks

This example [accesses a volume annotation on webKnososs](../../api/webknossos/annotation/annotation.md#Annotation.open_as_remote_dataset) and creates segment masks from specified segment IDs. The segment masks are stored as tiff sequences. The [BufferedSliceReader](../../api/webknossos/dataset/view.md#View.get_buffered_slice_reader) is used to efficiently request sections from the remote segmentation data.

```python
--8<--
webknossos/examples/download_segments.py
--8<--
```
