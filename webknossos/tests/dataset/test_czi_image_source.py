import importlib
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from upath import UPath

pytest.importorskip("pylibCZIrw")

from tests.utils import create_synthetic_czi  # noqa: E402
from webknossos.dataset import UnsupportedImageDataError  # noqa: E402
from webknossos.dataset._utils.czi_image_source import CziImageSource  # noqa: E402
from webknossos.dataset._utils.image_source import ReadOptions  # noqa: E402


def _open_czi_image_source(path: UPath, **options: Any) -> CziImageSource:
    return CziImageSource(path, ReadOptions(**options))


def test_czi_image_source_metadata(tmp_upath: UPath) -> None:
    # CZI reports its extents in metadata, so the bounding box is exact from
    # the start — there is no placeholder for the caller to correct.
    czi_path = tmp_upath / "meta.czi"
    create_synthetic_czi(czi_path, num_timepoints=1, num_czi_channels=2, z=4, y=8, x=10)

    source = _open_czi_image_source(czi_path)

    assert source.dtype == np.dtype("uint16")
    # A CZI "C" holds separate acquisitions, so it is offered as a layer split
    # rather than written as colour channels: one channel per layer.
    assert source.num_channels == 1
    assert source.get_possible_layers() == {"czi_channel": [0, 1]}
    assert source.expected_bbox.size.to_tuple() == (10, 8, 4)


def test_czi_image_source_multi_timepoint_gets_a_t_axis(tmp_upath: UPath) -> None:
    czi_path = tmp_upath / "multi_t.czi"
    create_synthetic_czi(czi_path, num_timepoints=3, num_czi_channels=1, z=2, y=8, x=10)

    source = _open_czi_image_source(czi_path)

    assert source.get_possible_layers() is None
    assert source.expected_bbox.axes == ("t", "x", "y", "z")
    assert source.expected_bbox.size.to_tuple() == (3, 10, 8, 2)


def test_czi_image_source_reads_only_the_requested_box(tmp_upath: UPath) -> None:
    # The point of reading CZI as a chunked source: a chunk must fetch just its
    # own box via pylibCZIrw's roi, not decode whole planes and crop. Without
    # the roi this still returns correct data, so only the read calls show it.
    from pylibCZIrw import czi as pyczi

    czi_path = tmp_upath / "roi.czi"
    data = create_synthetic_czi(czi_path, z=4, y=16, x=20)
    source = _open_czi_image_source(czi_path)

    rois = []
    original_read = pyczi.CziReader.read

    def tracking_read(self: Any, **kwargs: Any) -> np.ndarray:
        rois.append(kwargs.get("roi"))
        return original_read(self, **kwargs)

    with patch.object(pyczi.CziReader, "read", tracking_read):
        block = source._read_source_box(
            timepoint=0, z=slice(1, 3), y=slice(4, 12), x=slice(2, 10)
        )

    # one read per z-plane in the range, each asking for exactly that box
    assert rois == [(2, 4, 8, 8), (2, 4, 8, 8)]
    np.testing.assert_array_equal(block[0], data[0, 0, 1:3, 4:12, 2:10])


def test_czi_image_source_rejects_unsupported_dimension(tmp_upath: UPath) -> None:
    # CZI may carry rotation/illumination/phase/view/block dimensions. Pinning
    # one silently would drop data, so conversion refuses instead. pylibCZIrw's
    # writer cannot produce such a file, hence the stand-in reader.
    from pylibCZIrw import czi as pyczi

    czi_path = tmp_upath / "extra_dim.czi"
    create_synthetic_czi(czi_path)

    class _ReaderWithViewDimension:
        total_bounding_box = {
            "T": (0, 1),
            "Z": (0, 1),
            "C": (0, 1),
            "V": (0, 2),
            "X": (0, 10),
            "Y": (0, 8),
        }
        total_bounding_rectangle = pyczi.Rectangle(x=0, y=0, w=10, h=8)
        pixel_types = {0: "Gray16"}

        def get_channel_pixel_type(self, _channel: int) -> str:
            return "Gray16"

        def __enter__(self) -> "_ReaderWithViewDimension":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    czi_image_source = importlib.import_module(
        "webknossos.dataset._utils.czi_image_source"
    )
    with patch.object(
        czi_image_source.pyczi, "open_czi", lambda _path: _ReaderWithViewDimension()
    ):
        with pytest.raises(UnsupportedImageDataError, match="'V' dimension of size 2"):
            _open_czi_image_source(czi_path)
