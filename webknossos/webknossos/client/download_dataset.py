import warnings
from os import PathLike
from typing import List, Optional, Union

from webknossos.dataset import Dataset
from webknossos.geometry import BoundingBox, Mag

from ._download_dataset import download_dataset as actual_download_dataset


def download_dataset(
    dataset_name: str,
    organization_name: str,
    bbox: Optional[BoundingBox] = None,
    layers: Optional[List[str]] = None,
    mags: Optional[List[Mag]] = None,
    path: Optional[Union[PathLike, str]] = None,
    exist_ok: bool = False,
) -> Dataset:
    warnings.warn(
        "[DEPRECATION] download_dataset is deprecated, please use Dataset.download instead."
    )
    return actual_download_dataset(
        dataset_name, organization_name, bbox, layers, mags, path, exist_ok
    )
