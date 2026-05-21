import numpy as np
from pims import FramesSequenceND
from upath import UPath

try:
    import mrcfile
except ImportError as e:
    raise ImportError(
        "Cannot import mrcfile, please install it e.g. using 'webknossos[mrcfile]'"
    ) from e


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

        with mrcfile.mmap(str(self.path), mode="r", permissive=True) as mrc:
            if mrc.data is None:
                raise ValueError(
                    f"Cannot open MRC file {self.path}. "
                    + "The file is likely corrupted or not a valid MRC file."
                )
            self._dtype: np.dtype = mrc.data.dtype
            self._data_shape: tuple[int, ...] = tuple(mrc.data.shape)
            ndim = mrc.data.ndim

        if ndim == 3:
            nz, ny, nx = self._data_shape
            self._init_axis("z", nz)
            self._init_axis("y", ny)
            self._init_axis("x", nx)
            self._register_get_frame(self.get_frame_2D, "yx")
        elif ndim == 2:
            ny, nx = self._data_shape
            self._init_axis("y", ny)
            self._init_axis("x", nx)
            self._register_get_frame(self.get_frame_2D, "yx")
        else:
            raise ValueError(
                f"Unsupported MRC data dimensionality: {ndim}. "
                "Only 2D and 3D MRC files are supported."
            )

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        # Reopened per call so no mmap crosses a multiprocessing boundary.
        with mrcfile.mmap(str(self.path), mode="r", permissive=True) as mrc:
            if mrc.data.ndim == 3:
                return np.array(mrc.data[ind["z"]])
            return np.array(mrc.data[:])

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self._data_shape[-2:]
