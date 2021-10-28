# Dataset Usage

The high-level dataset API allows to interact with datasets while automatically maintaining meta data for any dataset,
such as the `datasource-properties.json`.

The [`Dataset` class](../../api/webknossos/dataset/dataset.md) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files.

Each dataset consists of one or more [layers](../../api/webknossos/dataset/layer.md),
which themselves can comprise multiple [magnifications represented via `MagView`s](../../api/webknossos/dataset/mag_view.md).

```python
--8<--
webknossos/examples/dataset_usage.py
--8<--
```
