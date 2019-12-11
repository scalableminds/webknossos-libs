import numpy as np
from wkw import Dataset

from wkcuber.api.TiffData.TiffMag import TiffMag, TiffMagHeader


class Slice:
    def __init__(
        self, path_to_mag_dataset, size=(1024, 1024, 1024), global_offset=(0, 0, 0)
    ):
        self.dataset = None
        self.path = path_to_mag_dataset
        self.size = size
        self.global_offset = global_offset
        self.is_bounded = True
        self._is_opened = False

    def open(self, header=None):
        raise NotImplemented()

    def close(self):
        if not self._is_opened:
            raise Exception("Cannot close slice: the slice is not opened")
        else:
            self.dataset.close()
            self.dataset = None
            self._is_opened = False

    def write(self, data, offset=(0, 0, 0), header=None):
        # assert the size of the parameter data is not in conflict with the attribute self.size
        self.assert_bounds(offset, data.shape[-3:])

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not self._is_opened:
            self.open(header)

        self.dataset.write(absolute_offset, data)

        if not self._is_opened:
            self.close()

    def read(self, size=None, offset=(0, 0, 0), header=None):
        was_opened = self._is_opened
        size = size or self.size

        # assert the parameter size is not in conflict with the attribute self.size
        self.assert_bounds(offset, size)

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not was_opened:
            self.open(header)

        data = self.dataset.read(absolute_offset, size)

        if not was_opened:
            self.close()

        return data

    def check_bounds(self, offset, size):
        for s1, s2, off in zip(self.size, size, offset):
            if s2 + off > s1 and self.is_bounded:
                return False
        return True

    def assert_bounds(self, offset, size):
        if not self.check_bounds(offset, size):
            raise Exception(
                "Writing out of bounds: The parameter 'size' {} is not compatible with the attribute 'size' {}".format(
                    size, self.size
                )
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


class WKSlice(Slice):
    def open(self, header=None):
        if self._is_opened:
            raise Exception("Cannot open slice: the slice is already opened")
        else:
            self.dataset = Dataset.open(self.path, header)
            self._is_opened = True
        return self

        """
        wkw.Header(
            convert_element_class_to_dtype(dtype),
            num_channels,
            file_len=args.wkw_file_len,
        ),
        """


class TiffSlice(Slice):
    def open(self, header=None):
        if self._is_opened:
            raise Exception("Cannot open slice: the slice is already opened")
        else:
            if header is None:
                header = TiffMagHeader()
            self.dataset = TiffMag.open(self.path, header)
            self._is_opened = True
        return self
