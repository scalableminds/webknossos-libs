import numpy as np
from wkw import Dataset

from wkcuber.api.TiffData.TiffMag import TiffMag


class View:
    def __init__(
        self,
        path_to_mag_dataset,
        header,
        size,
        global_offset=(0, 0, 0),
        is_bounded=True,
    ):
        self.dataset = None
        self.path = path_to_mag_dataset
        self.header = header
        self.size = size
        self.global_offset = global_offset
        self.is_bounded = is_bounded
        self._is_opened = False

    def open(self):
        raise NotImplemented()

    def close(self):
        if not self._is_opened:
            raise Exception("Cannot close View: the view is not opened")
        else:
            self.dataset.close()
            self.dataset = None
            self._is_opened = False

    def write(self, data, offset=(0, 0, 0)):
        # assert the size of the parameter data is not in conflict with the attribute self.size
        assert_non_negative_offset(offset)
        self.assert_bounds(offset, data.shape[-3:])

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not self._is_opened:
            self.open()

        self.dataset.write(absolute_offset, data)

        if not self._is_opened:
            self.close()

    def read(self, size=None, offset=(0, 0, 0)) -> np.array:
        was_opened = self._is_opened
        size = size or self.size

        # assert the parameter size is not in conflict with the attribute self.size
        self.assert_bounds(offset, size)

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not was_opened:
            self.open()

        data = self.dataset.read(absolute_offset, size)

        if not was_opened:
            self.close()

        return data

    def check_bounds(self, offset, size) -> bool:
        for s1, s2, off in zip(self.size, size, offset):
            if s2 + off > s1 and self.is_bounded:
                return False
        return True

    def assert_bounds(self, offset, size):
        if not self.check_bounds(offset, size):
            raise AssertionError(
                f"Writing out of bounds: The passed parameter 'size' {size} exceeds the size of the current view ({self.size})"
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


class WKView(View):
    def open(self):
        if self._is_opened:
            raise Exception("Cannot open view: the view is already opened")
        else:
            self.dataset = Dataset.open(
                self.path
            )  # No need to pass the header to the wkw.Dataset
            self._is_opened = True
        return self


class TiffView(View):
    def open(self):
        if self._is_opened:
            raise Exception("Cannot open view: the view is already opened")
        else:
            self.dataset = TiffMag.open(self.path, self.header)
            self._is_opened = True
        return self


def assert_non_negative_offset(offset):
    all_positive = all(i >= 0 for i in offset)
    if not all_positive:
        raise Exception(
            "All elements of the offset need to be positive: %s" % "("
            + ",".join(map(str, offset))
            + ")"
        )
