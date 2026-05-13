import numpy as np
from pims import FramesSequenceND
from upath import UPath

try:
    import mrcfile
except ImportError as e:
    raise ImportError(
        "Cannot import mrcfile, please install it e.g. using 'webknossos[mrcfile]'"
    ) from e

# MRC data mode → numpy dtype
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php
_MRC_MODE_TO_DTYPE: dict[int, np.dtype] = {
    0: np.dtype("int8"),
    1: np.dtype("int16"),
    2: np.dtype("float32"),
    6: np.dtype("uint16"),
    12: np.dtype("float16"),
}


class PimsMrcReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> set[str]:
        return {"mrc", "rec", "st", "map", "ali"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 19

    def __init__(self, path: UPath) -> None:
        super().__init__()
        self.path = UPath(path)

        # Use mrcfile only for header parsing — it handles endianness detection,
        # extended-header size, and header validation robustly.
        # We never touch mrc.data here: some files leave it None even though the
        # data bytes are perfectly valid (e.g. certain IMOD-written files).
        with mrcfile.mmap(str(self.path), mode="r", permissive=True) as mrc:
            mode = int(mrc.header.mode)
            if mode not in _MRC_MODE_TO_DTYPE:
                raise ValueError(
                    f"Unsupported MRC data mode {mode} in {self.path}. "
                    f"Supported modes: {list(_MRC_MODE_TO_DTYPE)}"
                )
            self._dtype = _MRC_MODE_TO_DTYPE[mode]
            nz = int(mrc.header.nz)
            ny = int(mrc.header.ny)
            nx = int(mrc.header.nx)
            # Standard MRC header is 1024 bytes; nsymbt is the extended header size.
            self._data_offset: int = 1024 + int(mrc.header.nsymbt)

        self._data_shape: tuple[int, ...] = (nz, ny, nx) if nz > 1 else (ny, nx)
        ndim = len(self._data_shape)

        if ndim == 3:
            self._init_axis("z", nz)
            self._init_axis("y", ny)
            self._init_axis("x", nx)
            self._register_get_frame(self.get_frame_2D, "yx")
        elif ndim == 2:
            self._init_axis("y", ny)
            self._init_axis("x", nx)
            self._register_get_frame(self.get_frame_2D, "yx")
        else:
            raise ValueError(
                f"Unsupported MRC data dimensionality: {ndim}. "
                "Only 2D and 3D MRC files are supported."
            )

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        # numpy.memmap is lazy (OS-paged) and cheap to create.
        # Recreated per call so no memmap crosses a multiprocessing boundary.
        data = np.memmap(
            str(self.path),
            dtype=self._dtype,
            mode="r",
            offset=self._data_offset,
            shape=self._data_shape,
        )
        if data.ndim == 3:
            return np.array(data[ind["z"]])
        return np.array(data[:])

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self._data_shape[-2:]
