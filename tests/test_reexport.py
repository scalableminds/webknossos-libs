from wkcuber import Mag, Dataset
from wkcuber.mag import Mag as _Mag
from wkcuber.api.dataset import Dataset as _Dataset


def test_reexport_classes() -> None:
    assert Mag == _Mag, "Mag exports should be the same class"
    assert Dataset == _Dataset, "Dataset exports should be the same class"
