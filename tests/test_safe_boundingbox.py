from wkcuber.SafeBoundingBox import VecNm
import numpy as np


def test_123():
    point = VecNm((10, 20, 40))
    assert np.array_equal(point.scaled((2, 2, 1)).scale, np.array((2, 2, 1)))
