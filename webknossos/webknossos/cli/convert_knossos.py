"""This module converts a KNOSSOS dataset into a WEBKNOSSOS dataset."""

import logging
import re
from argparse import Namespace
from collections import namedtuple
from collections.abc import Generator, Iterator
from functools import partial
from multiprocessing import cpu_count
from os import sep
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, cast

import numpy as np
import typer

from webknossos.dataset.length_unit import LengthUnit
from webknossos.dataset.properties import DEFAULT_LENGTH_UNIT_STR, VoxelSize

from ..dataset import COLOR_CATEGORY, DataFormat, Dataset, View
from ..dataset.defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import get_executor_for_args, time_start, time_stop
from ._utils import (
    DistributionStrategy,
    VoxelSizeTuple,
    parse_mag,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
    prepare_shard_shape,
)

KNOSSOS_CUBE_EDGE_LEN = 128
KNOSSOS_CUBE_SIZE = KNOSSOS_CUBE_EDGE_LEN**3
KNOSSOS_CUBE_SHAPE = (KNOSSOS_CUBE_EDGE_LEN,) * 3
KNOSSOS_CUBE_REGEX = re.compile(
    rf"x(\d+){re.escape(sep)}y(\d+){re.escape(sep)}z(\d+){re.escape(sep)}(.*\.raw)$"
)

KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))


class KnossosDataset:
    def __init__(self, root: str | Path, dtype: np.dtype):
        self.root = Path(root)
        self.dtype = dtype

    def read(
        self, offset: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> np.ndarray:
        assert offset[0] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert offset[1] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert offset[2] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert shape == KNOSSOS_CUBE_SHAPE
        return self.read_cube(tuple(x // KNOSSOS_CUBE_EDGE_LEN for x in offset))

    def write(self, offset: tuple[int, int, int], data: np.ndarray) -> None:
        assert offset[0] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert offset[1] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert offset[2] % KNOSSOS_CUBE_EDGE_LEN == 0
        assert data.shape == KNOSSOS_CUBE_SHAPE
        self.write_cube(tuple(x // KNOSSOS_CUBE_EDGE_LEN for x in offset), data)

    def read_cube(self, cube_xyz: tuple[int, ...]) -> np.ndarray:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            return np.zeros(KNOSSOS_CUBE_SHAPE, dtype=self.dtype)
        with open(filename, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=self.dtype)
            if cube_data.size != KNOSSOS_CUBE_SIZE:
                padded_data = np.zeros(KNOSSOS_CUBE_SIZE, dtype=self.dtype)
                padded_data[0 : min(cube_data.size, KNOSSOS_CUBE_SIZE)] = cube_data[
                    0 : min(cube_data.size, KNOSSOS_CUBE_SIZE)
                ]
                cube_data = padded_data
            cube_data = cube_data.reshape(KNOSSOS_CUBE_SHAPE, order="F")
            return cube_data

    def write_cube(self, cube_xyz: tuple[int, ...], cube_data: np.ndarray) -> None:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            filename = self.__get_cube_folder(cube_xyz) / self.__get_cube_file_name(
                cube_xyz
            )

        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)

    def __get_cube_folder(self, cube_xyz: tuple[int, ...]) -> Path:
        x, y, z = cube_xyz
        return self.root / f"x{x:04d}" / f"y{y:04d}" / f"z{z:04d}"

    def __get_cube_file_name(self, cube_xyz: tuple[int, ...]) -> Path:
        x, y, z = cube_xyz
        return Path(f"cube_x{x:04d}_y{y:04d}_z{z:04d}.raw")

    def __get_only_raw_file_path(self, cube_xyz: tuple[int, ...]) -> Path | None:
        cube_folder = self.__get_cube_folder(cube_xyz)
        raw_files = list(cube_folder.glob("*.raw"))
        assert len(raw_files) <= 1, (
            f"Found {len(raw_files)} .raw files in {cube_folder}"
        )
        return raw_files[0] if len(raw_files) > 0 else None

    def list_files(self) -> Iterator[Path]:
        return self.root.glob("*/*/*/*.raw")

    def __parse_cube_file_name(self, filename: Path) -> tuple[int, int, int] | None:
        m = KNOSSOS_CUBE_REGEX.search(str(filename))
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def list_cubes(self) -> Generator[tuple[int, int, int], Any, None]:
        return (
            f
            for f in (self.__parse_cube_file_name(f) for f in self.list_files())
            if f is not None
        )

    def close(self) -> None:
        pass

    @staticmethod
    def open(root: str | Path, dtype: np.dtype) -> "KnossosDataset":
        return KnossosDataset(root, dtype)

    def __enter__(self) -> "KnossosDataset":
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()


def open_knossos(info: KnossosDatasetInfo) -> KnossosDataset:
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def convert_cube_job(
    source_knossos_info: KnossosDatasetInfo, args: tuple[View, int]
) -> None:
    target_view, _ = args

    time_start(f"Converting of {target_view.bounding_box}")
    cube_size = cast(tuple[int, int, int], (KNOSSOS_CUBE_EDGE_LEN,) * 3)

    offset = target_view.bounding_box.in_mag(target_view.mag).topleft_xyz
    size = target_view.bounding_box.in_mag(target_view.mag).size_xyz
    buffer = np.zeros(size.to_tuple(), dtype=target_view.get_dtype())
    with open_knossos(source_knossos_info) as source_knossos:
        for x in range(0, size.x, KNOSSOS_CUBE_EDGE_LEN):
            for y in range(0, size.y, KNOSSOS_CUBE_EDGE_LEN):
                for z in range(0, size.z, KNOSSOS_CUBE_EDGE_LEN):
                    cube_data = source_knossos.read(
                        Vec3Int(offset + (x, y, z)).to_tuple(), cube_size
                    )
                    buffer[
                        x : (x + KNOSSOS_CUBE_EDGE_LEN),
                        y : (y + KNOSSOS_CUBE_EDGE_LEN),
                        z : (z + KNOSSOS_CUBE_EDGE_LEN),
                    ] = cube_data
    target_view.write(buffer)

    time_stop(f"Converting of {target_view.bounding_box}")


def convert_knossos(
    source_path: Path,
    target_path: Path,
    layer_name: str,
    dtype: str,
    voxel_size_with_unit: VoxelSize,
    data_format: DataFormat,
    chunk_shape: Vec3Int,  # in target-mag
    shard_shape: Vec3Int,
    mag: Mag = Mag(1),
    args: Namespace | None = None,
) -> None:
    """Performs the conversion of a KNOSSOS dataset to a WEBKNOSSOS dataset."""

    source_knossos_info = KnossosDatasetInfo(source_path, dtype)

    target_dataset = Dataset(
        target_path, voxel_size_with_unit=voxel_size_with_unit, exist_ok=True
    )
    target_layer = target_dataset.get_or_add_layer(
        layer_name,
        COLOR_CATEGORY,
        data_format=data_format,
        dtype_per_channel=dtype,
    )

    with open_knossos(source_knossos_info) as source_knossos:
        knossos_cubes = np.array(list(source_knossos.list_cubes()))
        if len(knossos_cubes) == 0:
            logging.error(
                "No input KNOSSOS cubes found. Make sure to pass the path which "
                "points to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
            )
            exit(1)

        min_xyz = knossos_cubes.min(axis=0) * KNOSSOS_CUBE_EDGE_LEN
        max_xyz = (knossos_cubes.max(axis=0) + 1) * KNOSSOS_CUBE_EDGE_LEN
        target_layer.bounding_box = BoundingBox(
            Vec3Int(min_xyz), Vec3Int(max_xyz - min_xyz)
        )

    target_mag = target_layer.get_or_add_mag(
        mag, chunk_shape=chunk_shape, shard_shape=shard_shape
    )

    with get_executor_for_args(args) as executor:
        target_mag.for_each_chunk(
            partial(convert_cube_job, source_knossos_info),
            chunk_shape=shard_shape * mag,
            executor=executor,
            progress_desc=f"Converting knossos layer {layer_name}",
        )


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your image data.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Target path to save your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    voxel_size: Annotated[
        VoxelSizeTuple,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma separated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VoxelSize",
            show_default=False,
        ),
    ],
    unit: Annotated[
        LengthUnit,
        typer.Option(
            help="The unit of the voxel size.",
        ),
    ] = DEFAULT_LENGTH_UNIT_STR,  # type:ignore
    dtype: Annotated[
        str, typer.Option(help="Target datatype (e.g. uint8, uint16, uint32)")
    ] = "uint8",
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = str(DEFAULT_DATA_FORMAT),  # type:ignore
    chunk_shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    shard_shape: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Number of voxels to be stored as a shard in the output format "
            "(e.g. `1024` or `1024,1024,1024`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    chunks_per_shard: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Deprecated, use --shard-shape. Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    mag: Annotated[
        Mag,
        typer.Option(
            help="Mag to start upsampling from. "
            "Should be number or minus separated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = 1,  # type: ignore
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        str | None,
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Convert your KNOSSOS dataset to a WEBKNOSOOS dataset."""

    shard_shape = prepare_shard_shape(
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        chunks_per_shard=chunks_per_shard,
    )

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    voxel_size_with_unit = VoxelSize(voxel_size, unit)

    convert_knossos(
        source,
        target,
        layer_name,
        dtype,
        voxel_size_with_unit,
        data_format,
        chunk_shape,
        shard_shape or DEFAULT_SHARD_SHAPE,
        mag,
        executor_args,
    )
