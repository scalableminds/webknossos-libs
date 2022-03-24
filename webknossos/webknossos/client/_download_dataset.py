import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from rich.progress import track

from webknossos.client._generated.api.datastore import dataset_download
from webknossos.client._generated.api.default import dataset_info
from webknossos.client.context import _get_context, _get_generated_client
from webknossos.dataset import Dataset, LayerCategoryType
from webknossos.geometry import BoundingBox, Mag, Vec3Int

logger = logging.getLogger(__name__)


T = TypeVar("T")


_DOWNLOAD_CHUNK_SIZE = Vec3Int(512, 512, 512)


def download_dataset(
    dataset_name: str,
    organization_id: Optional[str] = None,
    sharing_token: Optional[str] = None,
    bbox: Optional[BoundingBox] = None,
    layers: Optional[List[str]] = None,
    mags: Optional[List[Mag]] = None,
    path: Optional[Union[PathLike, str]] = None,
    exist_ok: bool = False,
) -> Dataset:
    client = _get_generated_client()
    context = _get_context()

    if organization_id is None:
        organization_id = context.organization_id

    if sharing_token is None:
        dataset_name_with_sharing_token = dataset_name
    else:
        dataset_name_with_sharing_token = f"{dataset_name}?sharingToken={sharing_token}"

    dataset_info_response = dataset_info.sync_detailed(
        organization_name=organization_id,
        data_set_name=dataset_name_with_sharing_token,
        client=client,
    )
    assert dataset_info_response.status_code == 200, dataset_info_response
    parsed = dataset_info_response.parsed
    assert parsed is not None

    datastore_client = context.get_generated_datastore_client(parsed.data_store.url)
    optional_datastore_token = sharing_token or context.datastore_token

    actual_path = Path(dataset_name) if path is None else Path(path)
    if actual_path.exists():
        logger.warning(f"{actual_path} already exists, skipping download.")
        return Dataset.open(actual_path)

    scale = cast(Tuple[float, float, float], tuple(parsed.data_source.scale))
    data_layers = parsed.data_source.data_layers
    dataset = Dataset(actual_path, name=parsed.name, scale=scale, exist_ok=exist_ok)
    for layer_name in layers or [i.name for i in data_layers]:

        response_layers = [i for i in data_layers if i.name == layer_name]
        assert (
            len(response_layers) > 0
        ), f"The provided layer name {layer_name} could not be found in the requested dataset."
        assert (
            len(response_layers) == 1
        ), f"The provided layer name {layer_name} was found multiple times in the requested dataset."
        response_layer = response_layers[0]
        category = cast(LayerCategoryType, response_layer.category)
        layer = dataset.add_layer(
            layer_name=layer_name,
            category=category,
            dtype_per_layer=response_layer.element_class,
            num_channels=3 if response_layer.element_class == "uint24" else 1,
            largest_segment_id=response_layer.additional_properties.get(
                "largestSegmentId", None
            ),
        )
        if bbox is None:
            response_bbox = response_layer.bounding_box
            layer.bounding_box = BoundingBox(
                response_bbox.top_left,
                (response_bbox.width, response_bbox.height, response_bbox.depth),
            )
        else:
            assert isinstance(
                bbox, BoundingBox
            ), f"Expected a BoundingBox object for the bbox parameter but got {type(bbox)}"
            layer.bounding_box = bbox
        if mags is None:
            mags = [Mag(mag) for mag in response_layer.resolutions]
        for mag in mags:
            mag_view = layer.get_or_add_mag(
                mag,
                compress=True,
                chunk_size=Vec3Int.full(32),
                chunks_per_shard=_DOWNLOAD_CHUNK_SIZE // 32,
            )
            aligned_bbox = layer.bounding_box.align_with_mag(mag, ceil=True)
            download_chunk_size_in_mag = _DOWNLOAD_CHUNK_SIZE * mag.to_vec3_int()
            for chunk in track(
                list(
                    aligned_bbox.chunk(
                        download_chunk_size_in_mag, download_chunk_size_in_mag
                    )
                ),
                description=f"Downloading layer={layer.name} mag={mag}",
            ):
                chunk_in_mag = chunk.in_mag(mag)
                response = dataset_download.sync_detailed(
                    organization_name=organization_id,
                    data_set_name=dataset_name,
                    data_layer_name=layer_name,
                    resolution=mag.max_dim_log2,
                    client=datastore_client,
                    token=optional_datastore_token,
                    x=chunk.topleft.x,
                    y=chunk.topleft.y,
                    z=chunk.topleft.z,
                    width=chunk_in_mag.size.x,
                    height=chunk_in_mag.size.y,
                    depth=chunk_in_mag.size.z,
                )
                assert response.status_code == 200, response
                assert (
                    response.headers["missing-buckets"] == "[]"
                ), f"Download contained missing buckets {response.headers['missing-buckets']}."
                data = np.frombuffer(
                    response.content, dtype=layer.dtype_per_channel
                ).reshape(layer.num_channels, *chunk_in_mag.size, order="F")
                mag_view.write(data, absolute_offset=chunk.topleft)
    return dataset
