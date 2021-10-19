import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from rich.progress import track

from webknossos.client.context import get_generated_client
from webknossos.client.generated import Client as GeneratedClient
from webknossos.client.generated.api.datastore import dataset_download
from webknossos.client.generated.api.default import dataset_info
from webknossos.client.generated.types import Unset
from webknossos.dataset import Dataset
from webknossos.geometry import BoundingBox, Mag

logger = logging.getLogger(__name__)


T = TypeVar("T")


def no_unset(x: Union[Unset, T]) -> T:
    assert not isinstance(x, Unset)
    return x


_DOWNLOAD_CHUNK_SIZE = (512, 512, 512)


def download_dataset(
    dataset_name: str,
    organization_name: str,
    bbox: Optional[BoundingBox] = None,
    layers: Optional[List[str]] = None,
    mags: Optional[List[Mag]] = None,
    path: Optional[Union[PathLike, str]] = None,
) -> Dataset:
    client = get_generated_client()
    dataset_info_response = dataset_info.sync_detailed(
        organization_name=organization_name,
        data_set_name=dataset_name,
        client=client,
    )
    assert dataset_info_response.status_code == 200
    parsed = dataset_info_response.parsed
    assert parsed is not None

    actual_path = Path(dataset_name) if path is None else Path(path)
    if actual_path.exists():
        logger.warning(f"{actual_path} already exists, skipping download.")
        return Dataset(actual_path)

    data_source = no_unset(parsed.data_source)
    data_layers = no_unset(data_source.data_layers)
    scale = cast(Tuple[float, float, float], data_source.scale)
    dataset = Dataset.create(actual_path, name=no_unset(parsed.name), scale=scale)
    GeneratedClient(base_url=no_unset(no_unset(parsed.data_store).url))
    for layer_name in layers or [no_unset(i.name) for i in data_layers]:

        response_layer = [i for i in data_layers if i.name == layer_name][0]
        layer = dataset.add_layer(
            layer_name=layer_name,
            category=no_unset(response_layer.category),
            dtype_per_layer=no_unset(response_layer.element_class),
            num_channels=3 if response_layer.element_class == "uint24" else 1,
            largest_segment_id=response_layer.additional_properties.get(
                "largestSegmentId", None
            ),
        )
        if bbox is None:
            response_bbox = no_unset(response_layer.bounding_box)
            layer.bounding_box = BoundingBox(
                no_unset(response_bbox.top_left),
                cast(
                    Tuple[int, int, int],
                    (response_bbox.width, response_bbox.height, response_bbox.depth),
                ),
            )
        else:
            layer.bounding_box = bbox
        if mags is None:
            mags = [Mag(mag) for mag in no_unset(response_layer.resolutions)]
        for mag in mags:
            aligned_bbox = layer.bounding_box.align_with_mag(mag, ceil=True)
            for chunk in track(
                list(aligned_bbox.chunk(_DOWNLOAD_CHUNK_SIZE, _DOWNLOAD_CHUNK_SIZE)),
                description=f"Downloading {layer.name} layer",
            ):
                aligned_chunk_in_mag = chunk.in_mag(mag)
                response = dataset_download.sync_detailed(
                    organization_name=organization_name,
                    data_set_name=dataset_name,
                    data_layer_name=layer_name,
                    resolution=mag.max_dim_log2,
                    client=client,
                    x=aligned_chunk_in_mag.topleft.x,
                    y=aligned_chunk_in_mag.topleft.y,
                    z=aligned_chunk_in_mag.topleft.z,
                    width=aligned_chunk_in_mag.size.x,
                    height=aligned_chunk_in_mag.size.y,
                    depth=aligned_chunk_in_mag.size.z,
                )
                assert response.status_code == 200
                data = np.frombuffer(
                    response.content, dtype=layer.dtype_per_channel
                ).reshape(layer.num_channels, *aligned_chunk_in_mag.size, order="F")
                mag_view = layer.get_or_add_mag(
                    mag,
                    compress=True,
                    block_len=32,
                    file_len=_DOWNLOAD_CHUNK_SIZE[0] // 32,
                )
                mag_view.write(data, offset=aligned_chunk_in_mag.topleft)
    return dataset
