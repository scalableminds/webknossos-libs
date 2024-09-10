# Zarr and Dask interoperability

This example shows how to access the underlying [Zarr](https://zarr.dev) array of a [remote dataset](../../api/webknossos/dataset/dataset.md#webknossos.dataset.dataset.RemoteDataset). Accessing the Zarr array allows to use other libraries, such as [Dask](https://www.dask.org/) for parallel processing.

```python
--8<--
webknossos/examples/zarr_and_dask.py
--8<--
```
