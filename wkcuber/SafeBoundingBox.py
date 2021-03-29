from typing import Any, Optional, overload

import numpy as np
from copy import copy
from wkcuber.mag import Mag


class VecNm:
    def __init__(self, point: Any) -> None:
        self.a = np.array(point, dtype=np.float32)
        assert self.a.shape == (3,), "The shape of the point must have a length of 3"

    def scaled(self, scale: Any) -> "VecNmWithScale":
        return VecNmWithScale(self.a, scale)


class VecNmWithScale(VecNm):
    def __init__(self, point: Any, scale: Any):
        super().__init__(point)
        self.scale = np.array(scale, dtype=np.float32)
        assert self.scale.shape == (
            3,
        ), "The shape of the scale must have a length of 3"

    def test(self) -> None:
        print("test")


class VecAnyMag:
    def __init__(self, point: Any):
        self.a = np.array(point, dtype=np.int32)
        assert self.a.shape == (3,), "The shape of the point must have a length of 3"

    def __add__(self, other: "VecAnyMag") -> "VecAnyMag":
        return VecAnyMag(self.a + other.a)

    def __sub__(self, other: "VecAnyMag") -> "VecAnyMag":
        return VecAnyMag(self.a - other.a)

    def __mul__(self, factor: int) -> "VecAnyMag":
        instance = copy(self)
        instance.a *= factor
        return instance

    def __floordiv__(self, factor: int) -> "VecAnyMag":
        instance = copy(self)
        instance.a //= factor
        return instance


class VecAnyMagWithMag(VecAnyMag):
    def __init__(self, point: Any, mag: Any):
        super().__init__(point)
        self.mag = Mag(mag)

    def to_mag(self, target_mag: Any, floor: bool = True) -> "VecAnyMagWithMag":
        target_mag = Mag(target_mag)
        factor = target_mag.as_np() / self.mag.as_np()
        if floor:
            return VecAnyMagWithMag(np.array(self.a) // factor, target_mag)
        else:
            return VecAnyMagWithMag(-(-np.array(self.a) // factor), target_mag)

    def __add__(self, other: "VecAnyMagWithMag"):
        assert self.mag == other.mag
        return VecAnyMagWithMag(self.a + other.a, self.mag)

    def __sub__(self, other: "VecAnyMagWithMag"):
        assert self.mag == other.mag
        return VecAnyMagWithMag(self.a - other.a, self.mag)


class VecMag1(VecAnyMagWithMag):
    def __init__(self, point: Any):
        super().__init__(point, 1)


if __name__ == "__main__":
    b = VecNm((10, 20, 30))

    a = VecNmWithScale((10, 20, 30), scale=(2, 2, 1))
    a.test()

    b.scaled((2, 2, 1)).test()

    b.test()  # fails
    print("end")
