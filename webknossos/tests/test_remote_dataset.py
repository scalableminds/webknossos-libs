import numpy as np
from upath import UPath as Path

from webknossos.dataset import COLOR_CATEGORY, Dataset
from webknossos.dataset._array import DataFormat

S3_KEY = "ANTN35UAENTS5UIAEATD"
S3_SECRET = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ENDPOINT = "http://localhost:9000"


def test_s3_dataset() -> None:
    import s3fs  # pylint: disable=unused-import

    bucket = Path(
        "s3://test",
        key=S3_KEY,
        secret=S3_SECRET,
        client_kwargs={"endpoint_url": S3_ENDPOINT},
    )
    bucket.fs.mkdirs("test", exist_ok=True)

    ds_path = bucket / "ds1"
    if ds_path.exists():
        ds_path.rmdir()

    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr)
    mag1 = layer.add_mag(1)

    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag1.write(data_to_write)

    assert np.array_equal(
        data_to_write,
        mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )
