import json
import os
from typing import cast

import numpy as np
import pytest
from jsonschema import validate
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS_AND_OUTPUT_PATHS,
    assure_exported_properties,
    copy_simple_dataset,
    prepare_dataset_path,
)
from webknossos.dataset import (
    Dataset,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.defaults import (
    DEFAULT_DATA_FORMAT,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    AffineCoordinateTransformation,
    DataFormat,
    DatasetProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
    SegmentationLayerProperties,
    ThinPlateSplineCoordinateTransformation,
)
from webknossos.dataset_properties.structuring import get_dataset_converter
from webknossos.geometry import (
    Mag,
    NDBoundingBox,
)
from webknossos.geometry.constants import (
    C_AXIS,
    T_AXIS,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
)
from webknossos.utils import (
    copytree,
    snake_to_camel_case,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
def test_ome_ngff_0_4_metadata(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr, output_path)
    ds = Dataset(ds_path, voxel_size=(11, 11, 28))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr)
    layer.add_mag("1")
    layer.add_mag("2-2-1")

    assert (ds_path / ".zgroup").exists()
    assert (ds_path / "color" / ".zgroup").exists()
    assert (ds_path / "color" / ".zattrs").exists()
    assert (ds_path / "color" / "1" / ".zarray").exists()
    assert (ds_path / "color" / "2-2-1" / ".zarray").exists()

    zattrs = json.loads((ds_path / "color" / ".zattrs").read_bytes())
    assert len(zattrs["multiscales"][0]["datasets"]) == 2
    assert zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ] == [
        1,
        11,
        11,
        28,
    ]
    assert zattrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0][
        "scale"
    ] == [
        1,
        22,
        22,
        28,
    ]

    validate(
        instance=zattrs,
        schema=json.loads(
            UPath(
                "https://ngff.openmicroscopy.org/0.4/schemas/image.schema"
            ).read_bytes()
        ),
    )


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
def test_ome_ngff_0_5_metadata(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(ds_path, voxel_size=(11, 11, 28))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr3)
    layer.add_mag("1")
    layer.add_mag("2-2-1")

    assert (ds_path / "zarr.json").exists()
    assert (ds_path / "color" / "zarr.json").exists()
    assert (ds_path / "color" / "1" / "zarr.json").exists()
    assert (ds_path / "color" / "2-2-1" / "zarr.json").exists()

    zattrs = json.loads((ds_path / "color" / "zarr.json").read_bytes())["attributes"]
    assert zattrs["ome"]["version"] == "0.5"
    assert len(zattrs["ome"]["multiscales"][0]["datasets"]) == 2
    assert zattrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"] == [
        1,
        11,
        11,
        28,
    ]
    assert zattrs["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][
        0
    ]["scale"] == [
        1,
        22,
        22,
        28,
    ]

    validate(
        instance=zattrs,
        schema=json.loads(
            UPath(
                "https://ngff.openmicroscopy.org/0.5/schemas/image.schema"
            ).read_bytes()
        ),
    )


def test_ome_ngff_0_5_metadata_nd() -> None:
    """OME axes/scale must be derived from the layer's actual axes, not
    hard-coded to c,x,y,z, for a layer with an additional (e.g. time) axis.
    """
    ds_path = prepare_dataset_path(DataFormat.Zarr3, TESTOUTPUT_DIR, suffix="nd")
    ds = Dataset(ds_path, voxel_size=(11, 11, 28))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        data_format=DataFormat.Zarr3,
        bounding_box=NDBoundingBox(
            (0, 0, 0, 0, 0),
            (1, 10, 10, 10, 3),
            axes=(C_AXIS, X_AXIS, Y_AXIS, Z_AXIS, T_AXIS),
            index=(0, 1, 2, 3, 4),
        ),
    )
    layer.add_mag("1")

    zattrs = json.loads((ds_path / "color" / "zarr.json").read_bytes())["attributes"]
    axes = zattrs["ome"]["multiscales"][0]["axes"]
    assert [a["name"] for a in axes] == [C_AXIS, X_AXIS, Y_AXIS, Z_AXIS, T_AXIS]
    assert axes[-1] == {"name": T_AXIS, "type": "time"}
    assert zattrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"] == [1, 11, 11, 28, 1]

    mag_shape = json.loads((ds_path / "color" / "1" / "zarr.json").read_bytes())[
        "shape"
    ]
    assert len(axes) == len(mag_shape)

    validate(
        instance=zattrs,
        schema=json.loads(
            UPath(
                "https://ngff.openmicroscopy.org/0.5/schemas/image.schema"
            ).read_bytes()
        ),
    )


def test_ome_ngff_0_5_metadata_symlink() -> None:
    def recursive_chmod(ds_path: UPath, mode: int) -> None:
        from pathlib import Path

        # See https://docs.python.org/3/library/os.html#os.chmod for how to use mode
        pathlib_path = Path(str(ds_path))
        os.chmod(pathlib_path, mode)
        for root, dirs, files in os.walk(pathlib_path):
            root_path = Path(root)
            for _dir in dirs:
                path = root_path / _dir
                os.chmod(path, mode)
            for file in files:
                path = root_path / file
                os.chmod(path, mode)

    ds_path = copy_simple_dataset(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "original")
    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    # remove write permissions
    recursive_chmod(ds_path, 0o555)
    try:
        ref_path = prepare_dataset_path(
            DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "with_refs"
        )
        ds = Dataset(ref_path, voxel_size=(1, 1, 1))

        ds.add_layer_as_ref(ds_path / "color")

    finally:
        # restore write permissions
        recursive_chmod(ds_path, 0o777)


def test_no_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(10, 10, 10))

    ds.add_layer("segmentation", SEGMENTATION_CATEGORY).add_mag(Mag(1))

    ds = Dataset.open(ds_path)

    assert (
        ds.get_layer("segmentation").as_segmentation_layer().largest_segment_id is None
    )

    assure_exported_properties(ds)


def test_properties_with_segmentation() -> None:
    ds_path = prepare_dataset_path(
        DataFormat.WKW, TESTOUTPUT_DIR, "complex_property_ds"
    )
    copytree(TESTDATA_DIR / "complex_property_ds", ds_path)

    data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    ds_properties = get_dataset_converter().structure(data, DatasetProperties)

    # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
    segmentation_layer = cast(
        SegmentationLayerProperties,
        [layer for layer in ds_properties.data_layers if layer.name == "segmentation"][
            0
        ],
    )
    color_layer = [
        layer for layer in ds_properties.data_layers if layer.name == "color"
    ][0]
    assert color_layer.coordinate_transformations == [
        AffineCoordinateTransformation.from_translation((10, 20, 30))
    ]
    assert segmentation_layer.coordinate_transformations == [
        ThinPlateSplineCoordinateTransformation(
            source=[[0, 0, 0], [1, 2, 3], [4, 5, 6]],
            target=[[1, 1, 1], [2, 4, 6], [8, 10, 12]],
        )
    ]

    assert segmentation_layer.largest_segment_id == 1000000000
    assert segmentation_layer.mappings == [
        "larger5um1",
        "axons",
        "astrocyte-ge-7",
        "astrocyte",
        "mitochondria",
        "astrocyte-full",
    ]

    # Update the properties on disk (without changing the data)
    (ds_path / PROPERTIES_FILE_NAME).write_text(
        json.dumps(
            get_dataset_converter().unstructure(ds_properties),
            indent=4,
        )
    )

    # validate if contents match
    input_data = json.loads(
        (TESTDATA_DIR / "complex_property_ds" / PROPERTIES_FILE_NAME).read_text()
    )

    output_data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    for layer in output_data["dataLayers"]:
        # check that numChannels and axisOrder are present
        # but remove them for the full check because they were not part of the original json
        assert layer["numChannels"] == 1
        layer.pop("numChannels", None)
        for mag in layer["mags"]:
            assert mag["axisOrder"] == {C_AXIS: 0, X_AXIS: 1, Y_AXIS: 2, Z_AXIS: 3}
            mag.pop("axisOrder", None)

    assert input_data == output_data


def test_dataset_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, voxel_size=(2, 2, 1))
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is None

    ds1.default_view_configuration = DatasetViewConfiguration(four_bit=True)
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation is None
    assert default_view_configuration.render_missing_data_black is None
    assert default_view_configuration.loading_strategy is None
    assert default_view_configuration.segmentation_pattern_opacity is None
    assert default_view_configuration.zoom is None
    assert default_view_configuration.position is None
    assert default_view_configuration.rotation is None

    # Test if only the set parameters are stored in the properties
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
    assert properties["defaultViewConfiguration"] == {"fourBit": True}

    ds1.default_view_configuration = DatasetViewConfiguration(
        four_bit=True,
        interpolation=False,
        render_missing_data_black=True,
        loading_strategy="PROGRESSIVE_QUALITY",
        segmentation_pattern_opacity=40,
        zoom=0.1,
        position=(12, 12, 12),
        rotation=(1, 2, 3),
    )

    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation == False
    assert default_view_configuration.render_missing_data_black == True
    assert default_view_configuration.loading_strategy == "PROGRESSIVE_QUALITY"
    assert default_view_configuration.segmentation_pattern_opacity == 40
    assert default_view_configuration.zoom == 0.1
    assert default_view_configuration.position == (12, 12, 12)
    assert default_view_configuration.rotation == (1, 2, 3)

    # Test if the data is persisted to disk
    ds2 = Dataset.open(ds_path)
    default_view_configuration = ds2.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation == False
    assert default_view_configuration.render_missing_data_black == True
    assert default_view_configuration.loading_strategy == "PROGRESSIVE_QUALITY"
    assert default_view_configuration.segmentation_pattern_opacity == 40
    assert default_view_configuration.zoom == 0.1
    assert default_view_configuration.position == (12, 12, 12)
    assert default_view_configuration.rotation == (1, 2, 3)

    # Test camel case
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
    view_configuration_dict = properties["defaultViewConfiguration"]
    for k in view_configuration_dict.keys():
        assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_layer_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, voxel_size=(2, 2, 1))
    layer1 = ds1.add_layer("color", COLOR_CATEGORY)
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is None

    layer1.default_view_configuration = LayerViewConfiguration(color=(255, 0, 0))
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha is None
    assert default_view_configuration.intensity_range is None
    assert default_view_configuration.is_inverted is None
    # Test if only the set parameters are stored in the properties
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
    assert properties["dataLayers"][0]["defaultViewConfiguration"] == {
        "color": [255, 0, 0]
    }

    layer1.default_view_configuration = LayerViewConfiguration(
        color=(255, 0, 0),
        alpha=1.0,
        min=55.0,
        intensity_range=(-12.3e1, 123),
        is_inverted=True,
    )
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha == 1.0
    assert default_view_configuration.intensity_range == (-12.3e1, 123)
    assert default_view_configuration.is_inverted == True
    assert default_view_configuration.min == 55.0

    # Test if the data is persisted to disk
    ds2 = Dataset.open(ds_path)
    default_view_configuration = ds2.get_layer("color").default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha == 1.0
    assert default_view_configuration.intensity_range == (-12.3e1, 123)
    assert default_view_configuration.is_inverted == True
    assert default_view_configuration.min == 55.0

    # Test camel case
    properties = json.loads((ds2.path / PROPERTIES_FILE_NAME).read_text())
    view_configuration_dict = properties["dataLayers"][0]["defaultViewConfiguration"]
    for k in view_configuration_dict.keys():
        assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_get_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    segmentation_layer = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).as_segmentation_layer()
    assert segmentation_layer.largest_segment_id == 999
    segmentation_layer.largest_segment_id = 123
    assert segmentation_layer.largest_segment_id == 123

    assure_exported_properties(ds)


def _largest_segment_id_on_disk(ds: Dataset, layer_name: str) -> int | dict[str, str]:
    properties = json.loads((ds.path / PROPERTIES_FILE_NAME).read_text())
    layer = next(
        layer for layer in properties["dataLayers"] if layer["name"] == layer_name
    )
    return layer["largestSegmentId"]


def test_largest_segment_id_bigint_envelope() -> None:
    # Segment ids above 2**53 - 1 (the largest integer JS can represent exactly)
    # are written as {"customJsonEncoding": "bigint", "value": "<decimal string>"}
    # instead of a plain JSON number, see _bigint_envelope.py.
    js_max_safe_integer = 2**53 - 1
    huge_segment_id = 2**64 - 1  # full uint64 range

    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    segmentation_layer = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype="uint64",
        largest_segment_id=huge_segment_id,
    ).as_segmentation_layer()

    assert segmentation_layer.largest_segment_id == huge_segment_id
    assert _largest_segment_id_on_disk(ds, "segmentation") == {
        "customJsonEncoding": "bigint",
        "value": str(huge_segment_id),
    }
    assure_exported_properties(ds)

    # a value that still fits into a JS-safe integer keeps the plain number format
    segmentation_layer.largest_segment_id = js_max_safe_integer
    assert _largest_segment_id_on_disk(ds, "segmentation") == js_max_safe_integer
    assure_exported_properties(ds)

    # one above the threshold switches back to the envelope
    segmentation_layer.largest_segment_id = js_max_safe_integer + 1
    assert _largest_segment_id_on_disk(ds, "segmentation") == {
        "customJsonEncoding": "bigint",
        "value": str(js_max_safe_integer + 1),
    }
    assure_exported_properties(ds)


def test_read_largest_segment_id_bigint_envelope() -> None:
    # A datasource-properties.json written by a newer WEBKNOSSOS version may
    # already contain the envelope; the python client must understand it too.
    huge_segment_id = 2**64 - 1

    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, dtype="uint64", largest_segment_id=1
    )

    properties_path = ds.path / PROPERTIES_FILE_NAME
    properties = json.loads(properties_path.read_text())
    for layer in properties["dataLayers"]:
        if layer["name"] == "segmentation":
            layer["largestSegmentId"] = {
                "customJsonEncoding": "bigint",
                "value": str(huge_segment_id),
            }
    properties_path.write_text(json.dumps(properties))

    reopened_ds = Dataset.open(ds.path)
    segmentation_layer = reopened_ds.get_segmentation_layer("segmentation")
    assert segmentation_layer.largest_segment_id == huge_segment_id


def test_refresh_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    segmentation_layer = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY
    ).as_segmentation_layer()
    mag = segmentation_layer.add_mag(Mag(1))

    assert segmentation_layer.largest_segment_id is None

    write_data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    mag.write(data=write_data, allow_resize=True)

    segmentation_layer.refresh_largest_segment_id()

    assert segmentation_layer.largest_segment_id == np.max(write_data, initial=0)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_warn_outdated_properties(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds1 = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds2 = Dataset.open(ds_path)

    # Change ds1 and undo it again
    ds1.add_layer("color", COLOR_CATEGORY, data_format=data_format).add_mag(1)
    ds1.delete_layer("color")

    # Changing ds2 should work fine, since the properties on disk
    # haven't changed.
    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        data_format=data_format,
        largest_segment_id=1,
    ).add_mag(1)

    with pytest.raises(UserWarning):
        # Changing ds1 should raise a warning, since ds1
        # does not know about the change in ds2
        ds1.add_layer("color", COLOR_CATEGORY, data_format=data_format)


def test_dataset_properties_version() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    properties_path = ds.path / PROPERTIES_FILE_NAME
    properties = json.loads((properties_path).read_bytes())
    assert properties["version"] == 1

    # write invalid version
    properties["version"] = 9000
    properties_path.write_text(json.dumps(properties))

    with pytest.raises(AssertionError):
        Dataset.open(ds_path)
