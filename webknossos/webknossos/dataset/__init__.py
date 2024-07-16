"""
# Dataset API

The high-level dataset API automatically reads and writes meta information for any dataset and updates them if necessary, such as the `datasource-properties.json`.

A dataset (`webknossos.dataset.dataset.Dataset`) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files (see [WEBKNOSSOS-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap)).

Each dataset consists of one or more layers (webknossos.dataset.layer.Layer), which themselves can comprise multiple magnifications (webknossos.dataset.mag_view.MagView).
"""
# ruff: noqa: F401 imported but unused

from .data_format import DataFormat
from .dataset import Dataset, RemoteDataset
from .layer import Layer, SegmentationLayer
from .layer_categories import COLOR_CATEGORY, SEGMENTATION_CATEGORY, LayerCategoryType
from .length_unit import LengthUnit
from .mag_view import MagView
from .remote_folder import RemoteFolder
from .sampling_modes import SamplingModes
from .view import View
