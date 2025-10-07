import io
import json
import os
import re
import uuid
from argparse import Namespace
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile
from zlib import Z_BEST_SPEED

import attr
from cluster_tools import Executor, SequentialExecutor
from numpy._typing import DTypeLike
from upath import UPath
from zipp import Path as ZipPath

from ..cli._utils import DistributionStrategy
from ..dataset import (
    SEGMENTATION_CATEGORY,
    DataFormat,
    Dataset,
    Layer,
    SegmentationLayer,
)
from ..dataset._array import Zarr3Config
from ..dataset.defaults import PROPERTIES_FILE_NAME
from ..dataset.properties import DatasetProperties, dataset_converter
from ..geometry import Vec3Int
from ..utils import get_executor_for_args, is_fs_path

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]


MAG_RE = r"((\d+-\d+-)?\d+)"
SEP_RE = r"(\/|\\)"
CUBE_RE = rf"z\d+{SEP_RE}y\d+{SEP_RE}x\d+\.wkw"
ANNOTATION_WKW_PATH_RE = re.compile(rf"{MAG_RE}{SEP_RE}(header\.wkw|{CUBE_RE})")


@attr.define
class SegmentInformation:
    name: str | None
    anchor_position: Vec3Int | None
    color: Vector4 | None
    metadata: dict[str, str | int | float | Sequence[str]]


class VolumeLayerEditMode(Enum):
    """Defines the edit mode for volume layers."""

    TEMPORARY_DIRECTORY = "temporary_directory"  # Use a temporary directory for edits
    MEMORY = "memory"  # Use an in-memory store for edits


VOLUME_ANNOTATION_ZARR3_CONFIG = Zarr3Config(
    codecs=(
        {"name": "transpose", "configuration": {"order": "F"}},
        {
            "name": "bytes",
        },
        {
            "name": "blosc",
            "configuration": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "lz4",
                "shuffle": "shuffle",
                "typesize": 1,
            },
        },
    ),
    chunk_key_encoding={
        "name": "default",
        "configuration": {"separator": "."},
    },
)


@attr.define
class VolumeLayer:
    id: int
    name: str
    fallback_layer_name: str | None
    data_format: DataFormat
    zip: ZipPath | None
    segments: dict[int, SegmentInformation]
    largest_segment_id: int | None
    voxel_size: Vector3 | None
    dtype: DTypeLike | None = None
    layer_name: str = "volumeAnnotationData"

    def _default_zip_name(self) -> str:
        return f"data_{self.id}_{self.name}.zip"

    def _write_dir_to_zip(self, source: str) -> None:
        """
        Write all files from the given source directory into the volume layer's zip archive.

        Parameters:
            source: Path to the directory whose contents will be added to the zip archive.
        """
        assert self.zip is not None

        volume_zip_buffer = io.BytesIO()
        with ZipFile(
            volume_zip_buffer,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=Z_BEST_SPEED,
        ) as volume_layer_zipfile:
            for dirname, _, files in os.walk(source):
                for file in files:
                    full_path = os.path.join(dirname, file)
                    arcname = os.path.relpath(full_path, source)
                    if (
                        arcname == "zarr.json"
                        or arcname == self.layer_name + "/zarr.json"
                    ):
                        continue
                    volume_layer_zipfile.write(full_path, arcname)

        volume_zip_buffer.seek(0)
        with ZipFile(
            self.zip.root.filename,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=Z_BEST_SPEED,
        ) as annotation_zip:
            annotation_zip.writestr(self.zip.at, volume_zip_buffer.read())

        # updating self.zip.root.__lookup to include the new file
        self.zip = ZipPath(self.zip.root.filename, self.zip.at)
        assert self.zip.exists()

    @contextmanager
    def edit(
        self,
        *,
        edit_mode: VolumeLayerEditMode = VolumeLayerEditMode.TEMPORARY_DIRECTORY,
        executor: Executor | None = None,
    ) -> Generator[Layer | Any, None, None]:
        """
        Context manager to edit the volume layer.

        Args:
            edit_mode: Specifies the edit mode for the volume layer.
            executor: Optional executor for parallel rechunking.

        """

        if self.zip is None:
            raise ValueError(
                "VolumeLayer.zip is not specified but required for editing."
            )

        def _edit(
            dataset_path: UPath, executor: Executor | None = None
        ) -> Generator[Layer, None, None]:
            dataset = Dataset(dataset_path, voxel_size=self.voxel_size)
            assert self.zip is not None and self.zip.exists()

            if is_fs_path(dataset_path):
                segmentation_layer = self.export_to_dataset(
                    dataset, layer_name=self.layer_name
                )
            else:
                # copy to temporary directory first, as tensorstore cannot read from MemoryFileSystem
                with TemporaryDirectory() as tempdir:
                    temp_dataset = Dataset(tempdir, voxel_size=self.voxel_size)
                    temp_segmentation_layer = self.export_to_dataset(
                        temp_dataset, layer_name=self.layer_name
                    )
                    segmentation_layer = cast(
                        SegmentationLayer,
                        dataset.add_layer_as_copy(
                            foreign_layer=temp_segmentation_layer,
                        ),
                    )

            yield segmentation_layer

            with TemporaryDirectory() as rechunked_dir:
                for mag_view in segmentation_layer.mags.values():
                    mag_view.rechunk(
                        chunk_shape=mag_view.info.chunk_shape,
                        shard_shape=mag_view.info.chunk_shape,  # same as chunk_shape to disable sharding
                        compress=VOLUME_ANNOTATION_ZARR3_CONFIG,
                        _progress_desc=f"Compressing {mag_view.layer.name} {mag_view.name}",
                        executor=executor,
                        target_path=rechunked_dir,
                    )
                self._write_dir_to_zip(rechunked_dir)

        fallback_executor_args = Namespace(
            distribution_strategy=DistributionStrategy.SEQUENTIAL.value,
        )
        with get_executor_for_args(fallback_executor_args, executor) as executor:
            if edit_mode == VolumeLayerEditMode.TEMPORARY_DIRECTORY:
                with TemporaryDirectory() as tmp_dir:
                    return _edit(UPath(tmp_dir), executor)
            elif edit_mode == VolumeLayerEditMode.MEMORY:
                if not isinstance(executor, SequentialExecutor):
                    raise ValueError(
                        "In-memory editing only supports SequentialExecutor to avoid data"
                        " corruption due to concurrent writes."
                    )
                path = UPath(
                    f"edit_{self.id}_{self.name}_{uuid.uuid4()}.zip", protocol="memory"
                )
                try:
                    return _edit(path, executor)
                finally:
                    if path.exists():
                        path.rmdir(recursive=True)
            else:
                raise ValueError(f"Unsupported volume layer edit mode: {edit_mode}")

    def export_to_dataset(
        self,
        dataset: Dataset,
        layer_name: str = "volume_layer",
    ) -> SegmentationLayer:
        """Exports the volume layer to a dataset as a SegmentationLayer.

        Args:
            dataset: The target dataset to export to.
            layer_name: Name of the layer in the dataset.

        Returns:
            SegmentationLayer: The created segmentation layer.

        Raises:
            AssertionError: If the volume layer is not set up correctly.

        Examples:
            ```python
            # Export volume layer to dataset
            exported_layer = volume_layer.export_to_dataset(my_dataset, "my_volume_layer")
            ```
        """

        assert self.zip is not None, (
            "The selected volume layer data is not available and cannot be exported."
        )
        assert is_fs_path(dataset.path), (
            "The dataset path must be a local path to export the volume layer."
        )

        with self.zip.open(mode="rb") as f:
            data_zip = ZipFile(f)
            if self.data_format == DataFormat.WKW:
                wrong_files = [
                    i.filename
                    for i in data_zip.filelist
                    if ANNOTATION_WKW_PATH_RE.search(i.filename) is None
                ]
                assert len(wrong_files) == 0, (
                    f"The annotation contains unexpected files: {wrong_files}"
                )
                data_zip.extractall(dataset.path / layer_name)
                layer = cast(
                    SegmentationLayer,
                    dataset.add_layer_for_existing_files(
                        layer_name,
                        category=SEGMENTATION_CATEGORY,
                        largest_segment_id=self.largest_segment_id,
                    ),
                )
            elif self.data_format == DataFormat.Zarr3:
                datasource_properties = dataset_converter.structure(
                    json.loads(data_zip.read(PROPERTIES_FILE_NAME)), DatasetProperties
                )
                assert len(datasource_properties.data_layers) == 1, (
                    f"Volume data zip must contain exactly one layer, got {len(datasource_properties.data_layers)}"
                )
                layer_properties = datasource_properties.data_layers[0]
                internal_layer_name = layer_properties.name
                layer_properties.name = layer_name

                _extract_zip_folder(
                    data_zip, dataset.path / layer_name, f"{internal_layer_name}/"
                )

                layer = cast(
                    SegmentationLayer,
                    dataset._add_existing_layer(layer_properties),
                )

        if len(layer.mags) > 0:
            best_mag_view = layer.get_finest_mag()

            if self.largest_segment_id is None:
                max_value = max(
                    (
                        view.read().max()
                        for view in best_mag_view.get_views_on_disk(read_only=True)
                    ),
                    default=0,
                )
                layer.largest_segment_id = int(max_value)
            else:
                layer.largest_segment_id = self.largest_segment_id

        return layer


def _extract_zip_folder(zip_file: ZipFile, out_path: Path, prefix: str) -> None:
    for zip_entry in zip_file.filelist:
        if zip_entry.filename.startswith(prefix) and not zip_entry.is_dir():
            out_file_path = out_path / (zip_entry.filename[len(prefix) :])
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                zip_file.open(zip_entry, "r") as zip_f,
                out_file_path.open("wb") as out_f,
            ):
                copyfileobj(zip_f, out_f)
