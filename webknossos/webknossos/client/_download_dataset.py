import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, TypeVar, Union, cast

import numpy as np
from rich.progress import track

from webknossos.dataset.length_unit import length_unit_from_str

from ..dataset import Dataset, LayerCategoryType
from ..dataset.properties import LayerViewConfiguration, VoxelSize, dataset_converter
from ..geometry import BoundingBox, Mag, Vec3Int
from .context import _get_context

logger = logging.getLogger(__name__)


T = TypeVar("T")


_DOWNLOAD_CHUNK_SHAPE = Vec3Int(512, 512, 512)


def download_dataset(
    dataset_id: str,
    sharing_token: Optional[str] = None,
    bbox: Optional[BoundingBox] = None,
    layers: Optional[List[str]] = None,
    mags: Optional[List[Mag]] = None,
    path: Optional[Union[PathLike, str]] = None,
    exist_ok: bool = False,
) -> Dataset:
    context = _get_context()
    api_client = context.api_client
    api_dataset = api_client.dataset_info(dataset_id, sharing_token=sharing_token)
    organization_id = api_dataset.owning_organization
    directory_name = api_dataset.directory_name

    datastore_client = context.get_datastore_api_client(api_dataset.data_store.url)
    optional_datastore_token = sharing_token or context.datastore_token

    download_path = (
        Path(f"{api_dataset.name}-{api_dataset.id}") if path is None else Path(path)
    )
    if download_path.exists():
        logger.warning(f"{download_path} already exists, skipping download.")
        return Dataset.open(download_path)

    api_data_layers = api_dataset.data_source.data_layers
    scale = api_dataset.data_source.scale
    if api_data_layers is None or len(api_data_layers) == 0 or scale is None:
        raise RuntimeError(
            f"Could not download dataset {api_client.base_wk_url}/datasets/{api_dataset.id}: {api_dataset.data_source.status or 'Unknown error.'}"
        )
    dataset = Dataset(
        download_path,
        name=api_dataset.name,
        voxel_size_with_unit=VoxelSize(scale.factor, length_unit_from_str(scale.unit)),
        exist_ok=exist_ok,
    )
    for layer_name in layers or [i.name for i in api_data_layers]:
        matching_api_data_layers = [i for i in api_data_layers if i.name == layer_name]
        assert (
            len(matching_api_data_layers) > 0
        ), f"The provided layer name {layer_name} could not be found in the requested dataset."
        assert (
            len(matching_api_data_layers) == 1
        ), f"The provided layer name {layer_name} was found multiple times in the requested dataset."
        api_data_layer = matching_api_data_layers[0]
        category = cast(LayerCategoryType, api_data_layer.category)
        layer = dataset.add_layer(
            layer_name=layer_name,
            category=category,
            dtype_per_layer=api_data_layer.element_class,
            num_channels=3 if api_data_layer.element_class == "uint24" else 1,
            largest_segment_id=api_data_layer.largest_segment_id,
        )

        if api_data_layer.default_view_configuration is not None:
            default_view_configuration = dataset_converter.structure(
                api_data_layer.default_view_configuration, LayerViewConfiguration
            )
            layer.default_view_configuration = default_view_configuration

        if bbox is None:
            response_bbox = api_data_layer.bounding_box
            layer.bounding_box = BoundingBox(
                Vec3Int(response_bbox.top_left),
                Vec3Int(response_bbox.width, response_bbox.height, response_bbox.depth),
            )
        else:
            assert isinstance(
                bbox, BoundingBox
            ), f"Expected a BoundingBox object for the bbox parameter but got {type(bbox)}"
            layer.bounding_box = bbox
        if mags is None:
            mags = [Mag(mag) for mag in api_data_layer.resolutions]
        for mag in mags:
            mag_view = layer.get_or_add_mag(
                mag,
                compress=True,
                chunk_shape=Vec3Int.full(32),
                chunks_per_shard=_DOWNLOAD_CHUNK_SHAPE // 32,
            )
            aligned_bbox = layer.bounding_box.align_with_mag(mag, ceil=True)
            download_chunk_shape_in_mag = _DOWNLOAD_CHUNK_SHAPE * mag.to_vec3_int()
            for chunk in track(
                list(
                    aligned_bbox.chunk(
                        download_chunk_shape_in_mag, download_chunk_shape_in_mag
                    )
                ),
                description=f"Downloading layer={layer.name} mag={mag}",
            ):
                chunk_in_mag = chunk.in_mag(mag)
                chunk_bytes, missing_buckets = datastore_client.dataset_get_raw_data(
                    organization_id=organization_id,
                    directory_name=directory_name,
                    data_layer_name=layer_name,
                    mag=mag.to_long_layer_name(),
                    token=optional_datastore_token,
                    x=chunk.topleft.x,
                    y=chunk.topleft.y,
                    z=chunk.topleft.z,
                    width=chunk_in_mag.size.x,
                    height=chunk_in_mag.size.y,
                    depth=chunk_in_mag.size.z,
                )
                assert (
                    missing_buckets == "[]"
                ), f"Download contained missing buckets {missing_buckets}."
                data = np.frombuffer(
                    chunk_bytes, dtype=layer.dtype_per_channel
                ).reshape(layer.num_channels, *chunk_in_mag.size, order="F")
                mag_view.write(data, absolute_offset=chunk.topleft)
    return dataset
