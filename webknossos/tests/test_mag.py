import numpy as np

from webknossos.geometry import Mag


def test_mag_constructor() -> None:
    mag = Mag(16)
    assert mag.to_list() == [16, 16, 16]

    mag = Mag("256")
    assert mag.to_list() == [256, 256, 256]

    mag = Mag("16-2-4")

    assert mag.to_list() == [16, 2, 4]

    mag1 = Mag("16-2-4")
    mag2 = Mag("8-2-4")

    assert mag1 > mag2
    assert mag1.to_layer_name() == "16-2-4"

    assert np.all(mag1.to_np() == np.array([16, 2, 4]))
    assert mag1 == Mag(mag1)
    assert mag1 == Mag(mag1.to_np())
