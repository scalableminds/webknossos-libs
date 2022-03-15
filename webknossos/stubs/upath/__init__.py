from pathlib import Path
from typing import Any, Dict

import fsspec


class UPath(Path):
    fs: fsspec.AbstractFileSystem
    _kwargs: Dict[str, Any]
