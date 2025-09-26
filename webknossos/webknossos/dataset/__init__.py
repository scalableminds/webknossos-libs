"""
# Dataset API

The high-level dataset API automatically reads and writes meta information for any dataset and updates them if necessary, such as the `datasource-properties.json`.

A dataset (`webknossos.dataset.dataset.Dataset`) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files (see [WEBKNOSSOS-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap)).

Each dataset consists of one or more layers (webknossos.dataset.layer.Layer), which themselves can comprise multiple magnifications (webknossos.dataset.mag_view.MagView).
"""
# ruff: noqa: F401 imported but unused

from .attachments import (
    AgglomerateAttachment,
    Attachment,
    ConnectomeAttachment,
    CumsumAttachment,
    MeshAttachment,
    SegmentIndexAttachment,
)
from .dataset import Dataset
from .defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from .layer import Layer, SegmentationLayer
from .layer_to_link import LayerToLink
from .mag_view import MagView
from .remote_dataset import RemoteDataset
from .remote_folder import RemoteFolder
from .sampling_modes import SamplingModes
from .view import View
