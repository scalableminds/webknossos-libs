"""
# Dataset API

The high-level dataset API automatically reads and writes meta information for any dataset and updates them if necessary, such as the `datasource-properties.json`.

A dataset is the entry-point for this API. All datasets are subclassing the abstract `wkcuber.api.Dataset.AbstractDataset` class, which implements most of the functionality.

The following concrete implementations are available, differing in the way they store the data on disk:
- `wkcuber.api.Dataset.WKDataset` (for [webknossos-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap) datasets)
- `wkcuber.api.Dataset.TiffDataset`
- `wkcuber.api.Dataset.TiledTiffDataset`

Each dataset consists of one or more layers (wkcuber.api.Layer.Layer), which themselves can comprise multiple magnifications (wkcuber.api.MagDataset.MagDataset).
"""
