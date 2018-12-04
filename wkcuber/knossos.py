import numpy as np
import os
import re
from os import path
from glob import iglob, glob

CUBE_EDGE_LEN = 128
CUBE_REGEX = re.compile(r"x(\d+)/y(\d+)/z(\d+)/(.*\.raw)$")


class KnossosDataset:
    def __init__(self, root, dtype=np.uint8):
        self.root = root
        self.dtype = dtype

    def read(self, offset, shape):
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert shape[0] == CUBE_EDGE_LEN
        assert shape[1] == CUBE_EDGE_LEN
        assert shape[2] == CUBE_EDGE_LEN
        return self.read_cube(tuple(x // CUBE_EDGE_LEN for x in offset))

    def write(self, offset, data):
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert data.shape[0] == CUBE_EDGE_LEN
        assert data.shape[1] == CUBE_EDGE_LEN
        assert data.shape[2] == CUBE_EDGE_LEN
        self.write_cube(tuple(x // CUBE_EDGE_LEN for x in offset), data)

    def read_cube(self, cube_xyz):
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            return np.zeros((CUBE_EDGE_LEN,) * 3, dtype=self.dtype)
        with open(filename, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=self.dtype).reshape(
                (CUBE_EDGE_LEN,) * 3, order="F"
            )
            return cube_data

    def write_cube(self, cube_xyz, cube_data):
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            filename = path.join(
                self.__get_cube_folder(cube_xyz), self.__get_cube_file_name(cube_xyz)
            )
        os.makedirs(path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)

    def __get_cube_folder(self, cube_xyz):
        x, y, z = cube_xyz
        return path.join(
            self.root, "x{:04d}".format(x), "y{:04d}".format(y), "z{:04d}".format(z)
        )

    def __get_cube_file_name(self, cube_xyz):
        x, y, z = cube_xyz
        return "cube_x{:04d}_y{:04d}_z{:04d}.raw".format(x, y, z)

    def __get_only_raw_file_path(self, cube_xyz):
        raw_files = glob(path.join(self.__get_cube_folder(cube_xyz), "*.raw"))
        assert len(raw_files) <= 1, "Found %d .raw files in %s" % (
            len(raw_files),
            cube_folder,
        )
        return raw_files[0] if len(raw_files) > 0 else None

    def list_files(self):
        return iglob(path.join(self.root, "**", "*.raw"), recursive=True)

    def __parse_cube_file_name(self, filename):
        m = CUBE_REGEX.search(filename)
        if m is None:
            return None
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def list_cubes(self):
        return (
            f
            for f in (self.__parse_cube_file_name(f) for f in self.list_files())
            if f is not None
        )

    def close(self):
        pass

    @staticmethod
    def open(root: str, dtype):
        return KnossosDataset(root, dtype)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
