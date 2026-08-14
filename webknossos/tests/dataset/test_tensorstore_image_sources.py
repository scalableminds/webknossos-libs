"""Unit tests for ZarrImageSource, N5ImageSource and
NeuroglancerPrecomputedImageSource, constructed directly against small real
stores written with tensorstore itself (no monkeypatching needed, unlike
.ims — tensorstore can genuinely create every format tested here)."""

from collections.abc import Iterator
from warnings import catch_warnings, filterwarnings

import numpy as np
import pytest
from upath import UPath

from tests.utils import (
    create_synthetic_n5_dataset,
    create_synthetic_n5_pyramid,
    create_synthetic_neuroglancer_precomputed,
    create_synthetic_ome_zarr_multiscale,
    create_synthetic_zarr2_array,
    create_synthetic_zarr3_array,
)
from webknossos.dataset._utils.image_source import ReadOptions
from webknossos.dataset._utils.image_source_registry import (
    get_valid_chunked_reader_suffixes,
    is_chunked_source_directory,
    open_image_source,
)
from webknossos.dataset._utils.n5_image_source import N5ImageSource
from webknossos.dataset._utils.neuroglancer_precomputed_image_source import (
    NeuroglancerPrecomputedImageSource,
)
from webknossos.dataset._utils.zarr_image_source import ZarrImageSource
from webknossos.dataset.errors import CorruptImageError, UnsupportedImageFormatError


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with catch_warnings():
        filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_chunked_reader_suffixes_include_zarr_and_n5() -> None:
    assert {"zarr", "n5"} <= get_valid_chunked_reader_suffixes()


# --- Zarr: plain arrays -----------------------------------------------------


def test_zarr3_plain_array_with_dimension_names(tmp_upath: UPath) -> None:
    data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)  # z, y, x
    path = tmp_upath / "plain.zarr"
    create_synthetic_zarr3_array(path, data, dimension_names=["z", "y", "x"])

    source = ZarrImageSource(path, ReadOptions())
    assert source.dtype == np.uint8
    assert source.num_channels == 1
    assert source.get_possible_layers() is None
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (4, 3, 2)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 2), y=slice(0, 3), x=slice(0, 4)
    )
    np.testing.assert_array_equal(block[0], data)


def test_zarr3_plain_array_positional_fallback(tmp_upath: UPath) -> None:
    # No dimension_names -> guessed as (c, z, y, x) for a 4D array.
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    path = tmp_upath / "no_labels.zarr"
    create_synthetic_zarr3_array(path, data)

    source = ZarrImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (5, 4, 3)
    # raw_num_channels=2 -> pinned to channel 0, offered as a possible layer.
    assert source.num_channels == 1
    assert source.channel == 0
    assert source.get_possible_layers() == {"channel": [0, 1]}


def test_zarr2_plain_array(tmp_upath: UPath) -> None:
    # 2D data exercises the "no z axis at all" case: expected_bbox reports
    # z=1, and _read_source_box must synthesize that axis.
    data = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6)  # y, x
    path = tmp_upath / "plain_v2.zarr"
    create_synthetic_zarr2_array(path, data)

    source = ZarrImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (6, 5, 1)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 1), y=slice(0, 5), x=slice(0, 6)
    )
    assert block.shape == (1, 1, 5, 6)
    np.testing.assert_array_equal(block[0, 0], data)


# --- Zarr: OME-Zarr multiscale groups ---------------------------------------


def test_ome_zarr_05_picks_finest_regardless_of_list_order(tmp_upath: UPath) -> None:
    finest = np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8)
    coarse = finest[:, ::2, ::2]
    path = tmp_upath / "multiscale.zarr"
    # Coarse level listed first, on purpose.
    create_synthetic_ome_zarr_multiscale(
        path,
        [coarse, finest],
        zarr_version=3,
        scales=[[1.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        dataset_paths=["1", "0"],
    )

    source = ZarrImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (8, 8, 2)
    assert source.get_possible_layers() == {"scale": [0, 1]}
    block = source._read_source_box(
        timepoint=0, z=slice(0, 2), y=slice(0, 8), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest)


def test_ome_zarr_05_explicit_scale_selects_coarser_level(tmp_upath: UPath) -> None:
    finest = np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8)
    coarse = finest[:, ::2, ::2]
    path = tmp_upath / "multiscale.zarr"
    create_synthetic_ome_zarr_multiscale(path, [finest, coarse], zarr_version=3)

    source = ZarrImageSource(path, ReadOptions(format_options={"scale": 1}))
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y")) == (4, 4)


def test_ome_zarr_04_single_level_group_reports_no_scale_choice(
    tmp_upath: UPath,
) -> None:
    data = np.arange(2 * 6 * 6, dtype=np.uint8).reshape(2, 6, 6)
    path = tmp_upath / "single.zarr"
    create_synthetic_ome_zarr_multiscale(path, [data], zarr_version=2)

    source = ZarrImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (6, 6, 2)
    assert source.get_possible_layers() is None


def test_zarr_group_without_multiscales_is_unsupported(tmp_upath: UPath) -> None:
    path = tmp_upath / "not_ome.zarr"
    path.mkdir()
    (path / ".zgroup").write_text('{"zarr_format": 2}')
    with pytest.raises(UnsupportedImageFormatError):
        ZarrImageSource(path, ReadOptions())


def test_zarr_not_a_store_raises_corrupt(tmp_upath: UPath) -> None:
    path = tmp_upath / "empty_dir"
    path.mkdir()
    with pytest.raises(CorruptImageError):
        ZarrImageSource(path, ReadOptions())


def test_zarr_probe_directory_recognizes_bare_stores(tmp_upath: UPath) -> None:
    data = np.zeros((4, 4), dtype=np.uint8)
    path = tmp_upath / "bare_store"  # deliberately no .zarr suffix
    create_synthetic_zarr2_array(path, data)

    assert ZarrImageSource.probe_directory(path)
    assert is_chunked_source_directory(path)
    source = open_image_source(path, ReadOptions())
    assert isinstance(source, ZarrImageSource)


# --- N5 ----------------------------------------------------------------------


def test_n5_plain_dataset(tmp_upath: UPath) -> None:
    data = np.arange(2 * 5 * 6, dtype=np.uint16).reshape(2, 5, 6)  # z, y, x
    path = tmp_upath / "plain.n5"
    create_synthetic_n5_dataset(path, data)

    source = N5ImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (6, 5, 2)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 2), y=slice(0, 5), x=slice(0, 6)
    )
    np.testing.assert_array_equal(block[0], data)
    assert source.get_possible_layers() is None


def test_n5_pyramid_picks_finest_and_honours_explicit_scale(tmp_upath: UPath) -> None:
    finest = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    coarse = finest[:, ::2, ::2]
    path = tmp_upath / "pyramid.n5"
    create_synthetic_n5_pyramid(path, [finest, coarse])

    source = N5ImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y")) == (8, 8)
    assert source.get_possible_layers() == {"scale": [0, 1]}
    block = source._read_source_box(
        timepoint=0, z=slice(0, 2), y=slice(0, 8), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest)

    source2 = N5ImageSource(path, ReadOptions(format_options={"scale": 1}))
    bbox2 = source2.expected_bbox
    assert (bbox2.get_shape("x"), bbox2.get_shape("y")) == (4, 4)

    assert N5ImageSource.probe_directory(path)


def test_n5_pyramid_out_of_order_downsampling_factors_raises(tmp_upath: UPath) -> None:
    finest = np.zeros((2, 4, 4), dtype=np.uint8)
    coarse = np.zeros((2, 2, 2), dtype=np.uint8)
    path = tmp_upath / "bad_pyramid.n5"
    # s0's own downsamplingFactors claims it is coarser than s1 -- corrupt.
    create_synthetic_n5_pyramid(
        path,
        [finest, coarse],
        downsampling_factors=[[1, 4, 4], [1, 1, 1]],
    )
    with pytest.raises(CorruptImageError):
        N5ImageSource(path, ReadOptions())


def test_n5_not_a_dataset_raises_corrupt(tmp_upath: UPath) -> None:
    path = tmp_upath / "empty_dir"
    path.mkdir()
    with pytest.raises(CorruptImageError):
        N5ImageSource(path, ReadOptions())


# --- Neuroglancer precomputed --------------------------------------------


def test_neuroglancer_precomputed_picks_finest_regardless_of_list_order(
    tmp_upath: UPath,
) -> None:
    x, y, z, c = 8, 6, 2, 1
    finest = np.arange(x * y * z * c, dtype=np.uint8).reshape(x, y, z, c)
    coarse = finest[::2, ::2]
    path = tmp_upath / "precomputed"
    create_synthetic_neuroglancer_precomputed(
        path,
        [([4, 4, 4], coarse), ([2, 2, 2], finest)],  # coarse listed first
    )

    source = NeuroglancerPrecomputedImageSource(path, ReadOptions())
    bbox = source.expected_bbox
    assert (bbox.get_shape("x"), bbox.get_shape("y"), bbox.get_shape("z")) == (x, y, z)
    assert source.get_possible_layers() == {"scale": [0, 1]}
    block = source._read_source_box(
        timepoint=0, z=slice(0, z), y=slice(0, y), x=slice(0, x)
    )
    expected = np.moveaxis(finest[..., 0], (0, 1, 2), (2, 1, 0))[np.newaxis]
    np.testing.assert_array_equal(block, expected)


def test_neuroglancer_precomputed_probe_directory_and_dispatch(
    tmp_upath: UPath,
) -> None:
    x, y, z, c = 4, 4, 2, 1
    data = np.zeros((x, y, z, c), dtype=np.uint8)
    path = tmp_upath / "precomputed"
    create_synthetic_neuroglancer_precomputed(path, [([1, 1, 1], data)])

    assert NeuroglancerPrecomputedImageSource.probe_directory(path)
    assert is_chunked_source_directory(path)
    source = open_image_source(path, ReadOptions())
    assert isinstance(source, NeuroglancerPrecomputedImageSource)


def test_neuroglancer_precomputed_not_a_volume_raises_corrupt(
    tmp_upath: UPath,
) -> None:
    path = tmp_upath / "empty_dir"
    path.mkdir()
    with pytest.raises(CorruptImageError):
        NeuroglancerPrecomputedImageSource(path, ReadOptions())


def test_ordinary_image_directory_is_not_claimed_by_any_new_reader(
    tmp_upath: UPath,
) -> None:
    # A directory with no store marker file must keep falling through to
    # SlicedImageSource, unaffected by the new probe_directory hooks.
    path = tmp_upath / "images"
    path.mkdir()
    (path / "a.png").write_bytes(b"\x89PNG\r\n")
    assert not is_chunked_source_directory(path)
