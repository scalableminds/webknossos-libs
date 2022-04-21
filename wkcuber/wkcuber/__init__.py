"""
# wkcuber

This module provides CLI tools to modify or interact with datasets.
For information regarding the other functionalities, refer to the `README.md`.
"""

from webknossos import Mag, Dataset

from .compress import compress_mag
from .cubing import cubing
from .downsampling import downsample_mags
from .metadata import write_webknossos_metadata
