from pathlib import Path
from typing import Any, Dict


class UPath(Path):
    _kwargs: Dict[str, Any]
