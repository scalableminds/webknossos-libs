from wkcuber import Mag, WKDataset
from wkcuber.mag import Mag as _Mag
from wkcuber.api.Dataset import WKDataset as _WKDataset


def test_reexport_classes() -> None:
    assert Mag == _Mag, "Mag exports should be the same class"
    assert WKDataset == _WKDataset, "WKDataset exports should be the same class"
