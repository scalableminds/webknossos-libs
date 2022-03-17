"""
# Dataset API

The high-level dataset API automatically reads and writes meta information for any dataset and updates them if necessary, such as the `datasource-properties.json`.

A dataset (`webknossos.dataset.dataset.Dataset`) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files (see [webknossos-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap)).

Each dataset consists of one or more layers (webknossos.dataset.layer.Layer), which themselves can comprise multiple magnifications (webknossos.dataset.mag_view.MagView).
"""

from ._array import DataFormat
from .dataset import Dataset
from .layer import Layer, SegmentationLayer
from .layer_categories import COLOR_CATEGORY, SEGMENTATION_CATEGORY, LayerCategoryType
from .mag_view import MagView
from .view import View
