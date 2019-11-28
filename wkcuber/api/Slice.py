import numpy as np
from wkw import Dataset


class Slice:

    def __init__(self, path_to_mag_dataset, size=(1024, 1024, 1024), global_offset=(0, 0, 0)):
        self.dataset = None
        self.path = path_to_mag_dataset
        self.size = size
        self.global_offset = global_offset
        self.is_bounded = True
        self._is_opened = False

    def open(self):
        raise NotImplemented()

    def close(self):
        raise NotImplemented()

    def read(self, size=None, offset=(0, 0, 0)):
        pass

    def write(self, data, offset=(0, 0, 0)):
        pass

    def check_bounds(self, offset, size):
        is_out_of_bounds = True
        if self.is_bounded and is_out_of_bounds:
            raise IndexError("Cannot write outside of boundingbox")


class WKSlice(Slice):

    def open(self):  # TODO: this currently does not support header
        if self._is_opened:
            raise Exception("Cannot open slice: the slice is already opened")
        else:
            self.dataset = Dataset.open(self.path)
            self._is_opened = True

    def close(self):
        if not self._is_opened:
            raise Exception("Cannot close slice: the slice is not opened")
        else:
            self.dataset.close()
            self.dataset = None
            self._is_opened = False

    def write(self, data, offset=(0, 0, 0)):
        # data: ndarray
        # offset: triple

        # assert the size of the parameter data is not in conflict with the attribute self.size
        for s1, s2, off in zip(self.size, data.shape[-3:], offset):
            if s2 + off > s1 and self.is_bounded:
                raise Exception(
                    "The size parameter 'data' {} is not compatible with the attribute 'size' {}".format(data.shape[-3:], self.size))  # TODO: adjust the error msg

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not self._is_opened:
            self.open()

        self.dataset.write(absolute_offset, data)

        if not self._is_opened:
            self.close()

    def read(self, size=None, offset=(0, 0, 0)):
        # offset: triple
        # size: triple

        size = size or self.size

        # assert the parameter size is not in conflict with the attribute self.size
        for s1, s2, off in zip(self.size, size, offset):
            if s2 + off > s1 and self.is_bounded:
                raise Exception("The parameter 'size' {} is not compatible with the attribute 'size' {}".format(size, self.size))  # TODO: adjust the error msg

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not self._is_opened:
            self.open()

        data = self.dataset.read(absolute_offset, size)

        if not self._is_opened:
            self.close()

        return data


class TiffSlice(Slice):

    def write(self, data, offset=(0, 0, 0)):
        if not self._is_opened:
            self.open()

        # TOOD: actual write

        if not self._is_opened:
            self.close()

    def read(self, size=None, offset=(0, 0, 0)):

        size = size or self.size

        if not self._is_opened:
            self.open()

        # TOOD: actual read

        if not self._is_opened:
            self.close()

        return np.array()
