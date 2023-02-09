# Download datasets as TIFF sequences

This example [accesses a dataset on webKnososs](../../api/webknossos/dataset/dataset.md#Dataset.open_remote) and creates a TIFF sequences from the downloaded data. The [BufferedSliceReader](../../api/webknossos/dataset/view.md#View.get_buffered_slice_reader) is used to efficiently request sections from the remote data.

```python
--8<--
webknossos/examples/download_tiff_stack.py
--8<--
```
