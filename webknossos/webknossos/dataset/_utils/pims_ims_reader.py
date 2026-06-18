import h5py
import numpy as np
from pims import FramesSequenceND
from upath import UPath

from ...utils import WkImportError, is_remote_path

try:
    from imaris_ims_file_reader.ims import ims as ImsFile
except ImportError as e:
    raise WkImportError("imaris-ims-file-reader", "ims") from e


class PimsImsReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> set[str]:
        return {"ims"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 19

    def __init__(self, path: UPath) -> None:
        super().__init__()
        self.path = UPath(path)

        if is_remote_path(self.path):
            raise ValueError(
                f"Cannot open IMS file from {self.path}. The path must be a local file path."
            )

        # Open once to read metadata, then close — ImsFile holds an h5py file handle
        # which is not pickleable, so we must not retain it as an attribute.
        _ims = ImsFile(str(self.path), squeeze_output=False)
        t, c, z, y, x = _ims.shape
        self._file_shape: tuple[int, int, int, int, int] = (t, c, z, y, x)
        self._dtype: np.dtype = np.dtype(_ims.dtype)
        _ims.close()

        if t > 1:
            self._init_axis("t", t)
        if c > 1:
            self._init_axis("c", c)
        self._init_axis("z", z)
        self._init_axis("y", y)
        self._init_axis("x", x)

        if c > 1:
            self._register_get_frame(self.get_frame_2D, "cyx")
        else:
            self._register_get_frame(self.get_frame_2D, "yx")

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        t = ind.get("t", 0)
        z = ind.get("z", 0)
        _, c, _, y, x = self._file_shape

        # Reopen per call so no h5py handle crosses a multiprocessing boundary.
        with h5py.File(str(self.path), "r") as hf:
            dataset = hf["DataSet"]
            if "c" in self.bundle_axes:
                frames = []
                for ci in range(c):
                    loc = f"ResolutionLevel 0/TimePoint {t}/Channel {ci}/Data"
                    frames.append(dataset[loc][z])
                return np.stack(frames, axis=0)
            else:
                ci = ind.get("c", 0)
                loc = f"ResolutionLevel 0/TimePoint {t}/Channel {ci}/Data"
                return np.array(dataset[loc][z])

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._file_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        _, c, _, y, x = self._file_shape
        if c > 1:
            return (c, y, x)
        return (y, x)
