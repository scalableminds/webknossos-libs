# Remote Dataset Access

This example shows how to access [remote datasets](../../api/webknossos/dataset/remote_dataset.md#webknossos.dataset.RemoteDataset). This can be done directly using [RemoteDataset.open()](../../api/webknossos/dataset/remote_dataset.md#webknossos.dataset.RemoteDataset.open), or listing all available datasets via [RemoteDataset.list()](../../api/webknossos/dataset/remote_dataset.md#webknossos.dataset.RemoteDataset.list).

```python
--8<--
webknossos/examples/remote_datasets.py
--8<--
```

## Access Modes

The image data of a remote dataset can be reached in three ways, expressed by `RemoteAccessMode`:

| Mode | Description |
| --- | --- |
| `ZARR_STREAMING` | The WEBKNOSSOS datastore re-serves the data as Zarr. Works everywhere and is the default. It is also the only mode that can read annotations. |
| `PROXY_PATH` | The WEBKNOSSOS datastore proxies the bytes of the underlying storage, preserving its data format. |
| `DIRECT_PATH` | The underlying storage is read directly. This is the fastest option, but it requires that your machine has access to that storage (and credentials for it). |

`RemoteDataset.open(access_mode=...)` sets the default that all mags inherit. Individual mags can override it, so a dataset can be read directly where that works and through the datastore where it does not:

```python
import webknossos as wk

ds = wk.RemoteDataset.open("https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view")
layer = ds.get_layer("color")

mag1 = layer.get_mag(1, access_mode=wk.RemoteAccessMode.DIRECT_PATH)
mag2 = layer.get_mag(2, access_mode=wk.RemoteAccessMode.PROXY_PATH)
```

Each mag knows all of its available paths, regardless of how it is currently accessed, via `mag.paths`, a `dict[RemoteAccessMode, UPath]`. A mode is missing from the dict if it isn't available for that mag (e.g. `DIRECT_PATH` when the server doesn't expose it):

```python
mag = layer.get_mag(1)
print(mag.paths)
# {RemoteAccessMode.ZARR_STREAMING: ..., RemoteAccessMode.PROXY_PATH: ..., RemoteAccessMode.DIRECT_PATH: ...}
print(mag.paths.get(wk.RemoteAccessMode.DIRECT_PATH))  # e.g. s3://bucket/dataset/color/1, or None if not exposed
print(mag.data_format)  # what this mag actually serves
```

Only the direct path is stored in the dataset properties; the other paths in `mag.paths` are computed from the datastore URL.

Metadata (layer bounding boxes, view configurations, mags, attachments, ...) can be written back to the server under any access mode, as long as the dataset's properties stem from the WEBKNOSSOS api — which is the case unless the dataset is viewed through an annotation, or its data source is unusable. Reading and writing metadata is independent of which access mode is used to read the image data itself.
