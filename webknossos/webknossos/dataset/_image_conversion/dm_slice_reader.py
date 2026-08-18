from collections.abc import Generator
from contextlib import closing, contextmanager

import numpy as np
from upath import UPath

from .image_source_registry import register_slice_reader
from .slice_reader import SliceReader
from .vendor.dm3 import DM3  # type: ignore[attr-defined]
from .vendor.dm3 import dT_str as DM3_DTYPE_MAPPING  # type: ignore[attr-defined]
from .vendor.dm4 import DM4File  # type: ignore[attr-defined]


@register_slice_reader
class Dm3SliceReader(SliceReader):
    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"dm3"}

    # The highest class_priority wins among eligible readers. 10 is the
    # default; nothing else claims dm3.
    class_priority = 20

    def __init__(self, path: UPath) -> None:
        self.path = UPath(path)
        super().__init__()
        dm3_file = DM3(self.path)
        self._init_axis("x", dm3_file.width)
        self._init_axis("y", dm3_file.height)
        if dm3_file.depth > 1:
            self._init_axis("z", dm3_file.depth)
            self._set_get_slice(self._get_slice, "zyx")
        else:
            self._set_get_slice(self._get_slice, "yx")

    @property  # potential @cached_property for py3.8+
    def pixel_type(self) -> np.dtype:
        dm3_file = DM3(self.path)
        return np.dtype(DM3_DTYPE_MAPPING[dm3_file._data_type])

    def _get_slice(self, **ind: int) -> np.ndarray:
        del ind
        return DM3(self.path).imagedata


@register_slice_reader
class Dm4SliceReader(SliceReader):
    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"dm4"}

    # open_slice_reader() picks the eligible reader with the highest class_priority.
    # 10 is the default; nothing else claims dm4.
    class_priority = 20

    def __init__(self, path: UPath) -> None:
        self.path = UPath(path)
        super().__init__()
        with self.dm4_file() as dm4_file:
            tags = dm4_file.read_directory()
            image_data_tag = (
                tags.named_subdirs["ImageList"]
                .unnamed_subdirs[1]
                .named_subdirs["ImageData"]
            )
            self._image_tag = image_data_tag.named_tags["Data"]
            self._shape = [
                dm4_file.read_tag_data(i)
                for i in image_data_tag.named_subdirs["Dimensions"].unnamed_tags
            ]

            if len(self._shape) not in [2, 3]:
                raise ValueError(
                    f"DM4 file {self.path} has incompatible number of dimensions, got shape {self._shape}."
                )
            self._init_axis("x", self._shape[0])
            self._init_axis("y", self._shape[1])
            if len(self._shape) == 2:
                self._set_get_slice(self._get_slice, "yx")
            else:
                self._init_axis("z", self._shape[2])
                self._set_get_slice(self._get_slice, "zyx")

    @contextmanager
    def dm4_file(self) -> Generator[DM4File]:
        with closing(DM4File.open(self.path)) as dm4_file:
            yield dm4_file

    @property  # potential @cached_property for py3.8+
    def pixel_type(self) -> np.dtype:
        with self.dm4_file() as dm4_file:
            return np.dtype(dm4_file.read_tag_data_type(self._image_tag))

    def _get_slice(self, **ind: int) -> np.ndarray:
        del ind
        with self.dm4_file() as dm4_file:
            a = np.asarray(dm4_file.read_tag_data(self._image_tag))
        return a.reshape(self._shape[::-1])
