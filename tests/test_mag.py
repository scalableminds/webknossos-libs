import numpy as np
from wkcuber.mag import Mag
from wkcuber.metadata import detect_resolutions


def test_detect_resolutions():
    resolutions = sorted(list(detect_resolutions("testdata/WT1_wkw", "color")))
    assert [mag.to_layer_name() for mag in resolutions] == ["1", "2-2-1"]


def test_mag_constructor():
    mag = Mag(16)
    assert mag.to_array() == [16, 16, 16]

    mag = Mag("256")
    assert mag.to_array() == [256, 256, 256]

    mag = Mag("16-2-4")

    assert mag.to_array() == [16, 2, 4]

    mag1 = Mag("16-2-4")
    mag2 = Mag("8-2-4")

    assert mag1 > mag2
    assert mag1.to_layer_name() == "16-2-4"

    assert np.all(mag1.as_np() == np.array([16, 2, 4]))
    assert mag1 == Mag(mag1)
    assert mag1 == Mag(mag1.as_np())
