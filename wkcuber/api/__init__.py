"""
# High Level Dataset API

The high level Dataset API is a set of classes that encapsulate the main functionalities for interacting with a dataset.
The aim is to make the experience for the user easy as possible, by automatically reading important meta information
from the `datasource-properties.json` for any dataset and updating this file if necessary.

A dataset is the most high-level class of this API.
There are three types of datasets:
- `wkcuber.api.Dataset.WKDataset`
- `wkcuber.api.Dataset.TiffDataset`
- `wkcuber.api.Dataset.TiledTiffDataset`

Each of these datasets derives from `wkcuber.api.Dataset.WKDataset.AbstractDataset`, which implements most of the functionality.
The datasets differ in the way they store the data on disc.
In practice, most of the time `wkcuber.api.Dataset.WKDataset` is used because wkw-files are our go-to file format.
"""
