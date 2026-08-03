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

Each mag knows all three of its paths, regardless of how it is currently accessed:

```python
mag = layer.get_mag(1)
print(mag.direct_path)           # e.g. s3://bucket/dataset/color/1, or None if not exposed
print(mag.proxy_path)
print(mag.zarr_streaming_path)
print(mag.data_format)           # what this mag actually serves
```

Only `direct_path` is stored in the dataset properties; the other two are computed from the datastore URL.
