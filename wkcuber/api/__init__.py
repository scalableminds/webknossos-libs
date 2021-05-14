"""
# High Level Dataset API

The high level Dataset API is ...

There are three types of datasets:
- `wkcuber.api.Dataset.WKDataset`
- `wkcuber.api.Dataset.TiffDataset`
- `wkcuber.api.Dataset.TiledTiffDataset`

The datasets differ in the way they store the data on disc.
In practice, most of the time `wkcuber.api.Dataset.WKDataset` is used because wkw-files are our go-to file format.

`wkcuber.api.Dataset.WKDataset`
.. include:: ../../README.md
"""