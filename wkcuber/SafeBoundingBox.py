from abc import ABC
from typing import Any, Optional, overload

import numpy as np
from copy import copy

from wkcuber.mag import Mag


class AbstractVec(ABC):
    a: np.ndarray

    def __init__(self, point):
        pass

    def __getitem__(self, key):
        return self.a[key]

    def __add__(self, other: "AbstractVec") -> "AbstractVec":
        return type(self)(self.a + other.a)

    def __sub__(self, other: "VecMag") -> "VecMag":
        return type(self)(self.a - other.a)

    def __mul__(self, factor: int) -> "AbstractVec":
        instance = copy(self)
        instance.a *= factor
        return instance

    def __floordiv__(self, factor: int) -> "AbstractVec":
        instance = copy(self)
        instance.a //= factor
        return instance

    def __truediv__(self, factor: int) -> "AbstractVec":
        instance = copy(self)
        instance.a /= factor
        return instance


class VecMag(AbstractVec):
    def __init__(self, point: Any):
        self.a = np.array(point, dtype=np.int32)
        assert self.a.shape == (3,), "The shape of the point must have a length of 3"

    def with_mag(self, mag: Mag) -> "VecKnownMag":
        return VecKnownMag(self.a, mag)

    def as_nm(self, scale: Any) -> "VecNm":
        return VecNm(tuple([a*s for a, s in zip(self.a, scale)]))


class VecKnownMag(VecMag):
    def __init__(self, point: Any, mag: Any):
        super().__init__(point)
        self.mag = Mag(mag)

    def to_mag(self, target_mag: Any, floor: bool = True) -> "VecKnownMag":
        target_mag = Mag(target_mag)
        factor = target_mag.as_np() / self.mag.as_np()
        if floor:
            return VecKnownMag(np.array(self.a) // factor, target_mag)
        else:
            return VecKnownMag(-(-np.array(self.a) // factor), target_mag)

    def __add__(self, other: "VecKnownMag") -> "VecKnownMag":
        if isinstance(other, type(self)):
            assert self.mag == other.mag
        return VecKnownMag(self.a + other.a, self.mag)

    def __sub__(self, other: "VecKnownMag") -> "VecKnownMag":
        if isinstance(other, type(self)):
            assert self.mag == other.mag
        return VecKnownMag(self.a - other.a, self.mag)

    def as_nm(self, scale: Any) -> "VecNm":
        return VecNm(tuple([a*s*m for a, s, m in zip(self.a, scale, self.mag.as_np())]))


class VecMag1(VecKnownMag):
    def __init__(self, point: Any):
        super().__init__(point, 1)


class VecNm(AbstractVec):
    def __init__(self, point: Any) -> None:
        self.a = np.array(point, dtype=np.float32)
        assert self.a.shape == (3,), "The shape of the point must have a length of 3"

    @overload
    def as_mag(self, scale: Any) -> VecMag1: ...

    @overload
    def as_mag(self, scale: Any, mag: Mag, floor: bool = True) -> VecKnownMag: ...

    def as_mag(self, scale: Any, mag: Optional[Mag] = None, floor: bool = True) -> VecKnownMag:
        scale = np.array(scale)
        vec_mag1 = VecMag1(self.a // scale)
        if mag is None:
            return vec_mag1
        else:
            return vec_mag1.to_mag(mag, floor)


if __name__ == "__main__":
    b = VecNm((10, 20, 30))

    #a = VecNmWithScale((10, 20, 30), scale=(2, 2, 1))
    #a.test()

    #b.scaled((2, 2, 1)).test()

    #b.test()  # fails
    print("end")
