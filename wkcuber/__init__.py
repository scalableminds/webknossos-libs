"""
# wkcuber

This module provides fundamental methods to modify or interact with datasets.
The documentation is currently limited to the Dataset API in `wkcuber.api`. For information regarding the other functionalities,
refer to the `README.md`.
"""

from .api.dataset import Dataset
from .cubing import cubing
from .downsampling import downsample_mags
from .compress import compress_mag
from .mag import Mag
from .metadata import write_webknossos_metadata
