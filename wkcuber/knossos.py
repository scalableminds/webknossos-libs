from types import TracebackType
from typing import Tuple, Any, Generator, Iterator, Optional, Type

import numpy as np
import os
import re
from os import path
from glob import iglob, glob

CUBE_EDGE_LEN = 128
CUBE_SIZE = CUBE_EDGE_LEN ** 3
CUBE_SHAPE = (CUBE_EDGE_LEN,) * 3
CUBE_REGEX = re.compile(r"x(\d+)/y(\d+)/z(\d+)/(.*\.raw)$")


class KnossosDataset:
    def __init__(self, root: str, dtype: np.dtype = np.uint8):
        self.root = root
        self.dtype = dtype

    def read(
        self, offset: Tuple[int, int, int], shape: Tuple[int, int, int]
    ) -> np.ndarray:
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert shape == CUBE_SHAPE
        return self.read_cube(tuple(x // CUBE_EDGE_LEN for x in offset))

    def write(self, offset: Tuple[int, int, int], data: np.ndarray) -> None:
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert data.shape == CUBE_SHAPE
        self.write_cube(tuple(x // CUBE_EDGE_LEN for x in offset), data)

    def read_cube(self, cube_xyz: Tuple[int, ...]) -> np.ndarray:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            return np.zeros(CUBE_SHAPE, dtype=self.dtype)
        with open(filename, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=self.dtype)
            if cube_data.size != CUBE_SIZE:
                padded_data = np.zeros(CUBE_SIZE, dtype=self.dtype)
                padded_data[0 : min(cube_data.size, CUBE_SIZE)] = cube_data[
                    0 : min(cube_data.size, CUBE_SIZE)
                ]
                cube_data = padded_data
            cube_data = cube_data.reshape(CUBE_SHAPE, order="F")
            return cube_data

    def write_cube(self, cube_xyz: Tuple[int, ...], cube_data: np.ndarray) -> None:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            filename = path.join(
                self.__get_cube_folder(cube_xyz), self.__get_cube_file_name(cube_xyz)
            )
        os.makedirs(path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)

    def __get_cube_folder(self, cube_xyz: Tuple[int, ...]) -> str:
        x, y, z = cube_xyz
        return path.join(
            self.root, "x{:04d}".format(x), "y{:04d}".format(y), "z{:04d}".format(z)
        )

    def __get_cube_file_name(self, cube_xyz: Tuple[int, ...]) -> str:
        x, y, z = cube_xyz
        return "cube_x{:04d}_y{:04d}_z{:04d}.raw".format(x, y, z)

    def __get_only_raw_file_path(self, cube_xyz: Tuple[int, ...]) -> Optional[str]:
        cube_folder = self.__get_cube_folder(cube_xyz)
        raw_files = glob(path.join(cube_folder, "*.raw"))
        assert len(raw_files) <= 1, "Found %d .raw files in %s" % (
            len(raw_files),
            cube_folder,
        )
        return raw_files[0] if len(raw_files) > 0 else None

    def list_files(self) -> Iterator[str]:
        return iglob(path.join(self.root, "*", "*", "*", "*.raw"), recursive=True)

    def __parse_cube_file_name(self, filename: str) -> Optional[Tuple[int, int, int]]:
        m = CUBE_REGEX.search(filename)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def list_cubes(self) -> Generator[Tuple[int, int, int], Any, None]:
        return (
            f
            for f in (self.__parse_cube_file_name(f) for f in self.list_files())
            if f is not None
        )

    def close(self) -> None:
        pass

    @staticmethod
    def open(root: str, dtype: Optional[np.dtype]) -> "KnossosDataset":
        return KnossosDataset(root, dtype)

    def __enter__(self) -> "KnossosDataset":
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self.close()
