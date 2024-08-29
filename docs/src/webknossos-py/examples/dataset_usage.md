# Dataset Usage

The high-level dataset API allows to interact with datasets while automatically maintaining meta data for any dataset,
such as the `datasource-properties.json`.

The [`Dataset` class](../../api/webknossos/dataset/dataset.md) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files.

Each dataset consists of one or more [layers](../../api/webknossos/dataset/layer.md),
which themselves can comprise multiple [magnifications represented via `MagView`s](../../api/webknossos/dataset/magview.md).

```python
--8<--
webknossos/examples/dataset_usage.py
--8<--
```

## Parallel Access of WEBKNOSSOS Datasets

Please consider these restrictions when accessing a WEBKNOSSOS dataset in a multiprocessing-context:

 - When writing shards in parallel, `json_update_allowed` should be set to `False` to disable the automatic update of the bounding box metadata. Otherwise, race conditions may happen. The user is responsible for updating the bounding box manually.
 - When writing to chunks in shards, one chunk may only be written to by one actor at any time.
 - When writing to compressed shards, one shard may only be written to by one actor at any time.
 - For Zarr datasets, parallel write access to shards is not allowed at all.
 - Reading in parallel without concurrent writes is fine.
