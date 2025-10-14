# Download segment masks

This example [accesses a volume annotation on webKnososs](../../api/webknossos/annotation/annotation.md#webknossos.annotation.Annotation.open_as_remote_dataset) and creates segment masks from selected segments. The segment masks are stored as tiff sequences. The segments have been selected from the "Segments" tab in WEBKNOSSOS. The [BufferedSliceReader](../../api/webknossos/dataset/layer/view/view.md#webknossos.dataset.View.get_buffered_slice_reader) is used to efficiently request sections from the remote segmentation data.

```python
--8<--
webknossos/examples/download_segments.py
--8<--
```
