from pathlib import Path
from shutil import rmtree, unpack_archive
from subprocess import check_call

import pytest
from webknossos import Dataset, Mag
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.utils import copytree

from .constants import TESTDATA_DIR

WEBKNOSSOS_TESTDATA_DIR = Path("../webknossos/testdata")


@pytest.fixture()
def tiff_mag_2_reference_path(tmp_path: Path) -> Path:
    (tmp_path / "tiff_mag_2_reference").mkdir(parents=True)
    unpack_archive(
        TESTDATA_DIR / "tiff_mag_2_reference.tar.gz",
        tmp_path / "tiff_mag_2_reference",
    )
    return tmp_path / "tiff_mag_2_reference"


@pytest.fixture(scope="session")
def sample_wkw_path() -> Path:
    ds_path = TESTDATA_DIR / "tiff_wkw"
    if ds_path.exists():
        rmtree(ds_path)
    check_call(
        [
            "python",
            "-m",
            "wkcuber.cubing",
            "--jobs",
            "2",
            "--voxel_size",
            "1,1,1",
            str(TESTDATA_DIR / "tiff"),
            str(ds_path),
        ]
    )
    copytree(
        TESTDATA_DIR / "tiff" / "datasource-properties.wkw-fixture.json",
        ds_path / PROPERTIES_FILE_NAME,
    )
    Dataset.open(ds_path).get_layer("color").downsample_mag(Mag(1), Mag(2))
    return ds_path


@pytest.fixture(scope="session")
def WT1_path() -> Path:
    ds_path = WEBKNOSSOS_TESTDATA_DIR / "WT1_wkw"
    if ds_path.exists():
        rmtree(ds_path)
    unpack_archive(
        WEBKNOSSOS_TESTDATA_DIR / "WT1_wkw.tar.gz",
        ds_path,
    )
    return ds_path
