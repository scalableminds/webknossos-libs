from typing import Any, Optional, overload

import numpy as np

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
        assert self.scale.shape == (3,), "The shape of the scale must have a length of 3"

    def test(self) -> None:
        print("test")


class VecAnyMag:
    def __init__(self, point: Any):
        self.a = np.array(point, dtype=np.int32)
        assert self.a.shape == (3,), "The shape of the point must have a length of 3"


class VecAnyMagWithMag(VecAnyMag):
    def __init__(self, point: Any, mag: Any):
        super().__init__(point)
        self.mag = Mag(mag)


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

