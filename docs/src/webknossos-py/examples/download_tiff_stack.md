# Download datasets as TIFF sequences

This example [accesses a dataset on webKnososs](../../api/webknossos/dataset/remote_dataset.md#webknossos.dataset.remote_dataset.RemoteDataset.open) and creates a TIFF sequences from the downloaded data. The [BufferedSliceReader](../../api/webknossos/dataset/layer/view/view.md#webknossos.dataset.layer.view.View.get_buffered_slice_reader) is used to efficiently request sections from the remote data.

```python
--8<--
webknossos/examples/download_tiff_stack.py
--8<--
```
