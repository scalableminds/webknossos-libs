import json
import wkw
import re
import logging
import numpy as np
from os import PathLike
from os.path import sep

from argparse import ArgumentParser
from typing import Optional, Tuple, Iterable, Generator, Union

from webknossos.dataset.layer import LayerCategoryType
from .mag import Mag
from typing import List
from .utils import add_verbose_flag, setup_logging, add_scale_flag
from pathlib import Path

WKW_CUBE_REGEX = re.compile(
    fr"z(\d+){re.escape(sep)}y(\d+){re.escape(sep)}x(\d+)(\.wkw)$"
)


def parse_cube_file_name(filename: Union[PathLike, str]) -> Tuple[int, int, int]:
    match = WKW_CUBE_REGEX.search(str(filename))
    if match is None:
        raise ValueError(f"Failed to parse cube file name {filename}")
    return int(match.group(3)), int(match.group(2)), int(match.group(1))


def get_datasource_path(dataset_path: Path) -> Path:
    return dataset_path / "datasource-properties.json"


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.", type=Path)

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)

    parser.add_argument("--refresh", "-r", default=False, action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compute_max_id",
        "-c",
        help="set to compute max id",
        default=False,
        action="store_true",
    )
    group.add_argument("--max_id", help="set max id of segmentation.", default=0)

    add_scale_flag(parser, required=False)
    add_verbose_flag(parser)

    return parser


def write_datasource_properties(
    dataset_path: Path, datasource_properties: dict
) -> None:
    datasource_properties_path = get_datasource_path(dataset_path)
    with open(
        datasource_properties_path, "wt", encoding="utf-8"
    ) as datasource_properties_file:
        json.dump(datasource_properties, datasource_properties_file, indent=2)


def read_datasource_properties(dataset_path: Path) -> dict:
    with open(
        get_datasource_path(dataset_path), "r", encoding="utf-8"
    ) as datasource_properties_file:
        return json.load(datasource_properties_file)


def write_webknossos_metadata(
    dataset_path: Path,
    name: str,
    scale: Tuple[float, float, float],
    max_id: int = 0,
    compute_max_id: bool = False,
    exact_bounding_box: Optional[dict] = None,
    view_configuration: Optional[dict] = None,
) -> None:
    """
    Creates a datasource-properties.json file with the specified properties
    for the given dataset path. Common layers are detected automatically.
    """
    if name == None:
        name = dataset_path.name

    # Generate a metadata file for webKnossos
    # Currently includes no source of information for team
    layers = list(
        detect_layers(
            dataset_path, max_id, compute_max_id, exact_bounding_box, view_configuration
        )
    )
    write_datasource_properties(
        dataset_path,
        {
            "id": {"name": name, "team": "<unknown>"},
            "dataLayers": layers,
            "scale": scale,
        },
    )


def refresh_metadata(
    wkw_path: Path,
    max_id: int = 0,
    compute_max_id: bool = False,
    exact_bounding_box: Optional[dict] = None,
    view_configuration: Optional[dict] = None,
) -> None:
    """
    Updates the datasource-properties.json file for a given dataset.
    Use this method if you added (or removed) layers and/or changed magnifications for
    existing layers.

    Raises an exception if the datasource-properties.json file does not exist, yet.
    In this case, use write_webknossos_metadata instead.
    """
    dataset_path = get_datasource_path(wkw_path)
    if not dataset_path.exists():
        raise Exception(
            "datasource-properties.json file could not be found. Please use write_webknossos_metadata to create it."
        )

    datasource_properties = read_datasource_properties(wkw_path)
    existing_layers_dict = {
        layer["name"]: layer for layer in datasource_properties["dataLayers"]
    }

    new_layers = list(
        detect_layers(
            wkw_path, max_id, compute_max_id, exact_bounding_box, view_configuration
        )
    )

    # Merge the freshly read layers with the existing layer information, so that information,
    # such as bounding boxes, are not lost for existing layers.
    # For existing layers, only the resolutions will be updated.
    merged_layers = []
    for new_layer in new_layers:
        layer_name = new_layer["name"]
        if layer_name in existing_layers_dict:
            existing_layer = existing_layers_dict[layer_name]
            # Update the resolutions
            existing_layer["wkwResolutions"] = new_layer["wkwResolutions"]
            merged_layers.append(existing_layer)
        else:
            merged_layers.append(new_layer)

    datasource_properties["dataLayers"] = merged_layers
    write_datasource_properties(wkw_path, datasource_properties)


def convert_element_class_to_dtype(elementClass: str) -> np.dtype:
    fallback_dtype = "uint8" if "uint" in elementClass else elementClass
    conversion_map = {
        "float": "float32",
        "double": "float64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
    }
    return np.dtype(conversion_map.get(elementClass, fallback_dtype))


def read_metadata_for_layer(
    wkw_path: Path, layer_name: str
) -> Tuple[dict, np.dtype, List[int], List[int]]:
    datasource_properties = read_datasource_properties(wkw_path)

    layers = datasource_properties["dataLayers"]
    layer_info = next(layer for layer in layers if layer["name"] == layer_name)

    dtype = convert_element_class_to_dtype(layer_info["elementClass"])
    bounding_box = layer_info["boundingBox"]
    origin = bounding_box["topLeft"]
    bounding_box = [
        bounding_box["width"],
        bounding_box["height"],
        bounding_box["depth"],
    ]

    return layer_info, dtype, bounding_box, origin


def convert_dtype_to_element_class(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    dtype_to_element_class_map = {
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }
    return dtype_to_element_class_map.get(dtype, str(dtype))


def detect_mag_path(
    dataset_path: Path, layer: str, mag: Mag = Mag(1)
) -> Optional[Path]:
    layer_path = dataset_path / layer / str(mag)
    if layer_path.exists():
        return layer_path
    layer_path = dataset_path / layer / mag.to_long_layer_name()
    if layer_path.exists():
        return layer_path
    return None


def detect_dtype(dataset_path: Path, layer: str, mag: Mag = Mag(1)) -> str:
    layer_path = detect_mag_path(dataset_path, layer, mag)
    if layer_path is not None:
        with wkw.Dataset.open(str(layer_path)) as dataset:
            voxel_size = dataset.header.voxel_type
            num_channels = dataset.header.num_channels
            if voxel_size == np.uint8 and num_channels > 1:
                return "uint" + str(8 * num_channels)
            else:
                return convert_dtype_to_element_class(voxel_size)
    raise RuntimeError(
        f"Failed to detect dtype (for {dataset_path}, {layer}, {mag}) because the layer_path is None"
    )


def detect_cubeLength(dataset_path: Path, layer: str, mag: Mag = Mag(1)) -> int:
    layer_path = detect_mag_path(dataset_path, layer, mag)
    if layer_path is not None:
        with wkw.Dataset.open(str(layer_path)) as dataset:
            return dataset.header.block_len * dataset.header.file_len
    raise RuntimeError(
        f"Failed to detect the cube length (for {dataset_path}, {layer}, {mag}) because the layer_path is None"
    )


def detect_bbox(dataset_path: Path, layer: str, mag: Mag = Mag(1)) -> Optional[dict]:
    # Detect the coarse bounding box of a dataset by iterating
    # over the WKW cubes
    layer_path = detect_mag_path(dataset_path, layer, mag)
    if layer_path is None:
        return None

    cubes_list = [parse_cube_file_name(f) for f in layer_path.rglob("*/*/*.wkw")]
    if len(cubes_list) == 0:
        return None

    xs, ys, zs = list(zip(*cubes_list))

    min_x, min_y, min_z = min(xs), min(ys), min(zs)
    max_x, max_y, max_z = max(xs), max(ys), max(zs)

    cubeLength = detect_cubeLength(dataset_path, layer, mag)
    if cubeLength is None:
        return None

    return {
        "topLeft": [min_x * cubeLength, min_y * cubeLength, min_z * cubeLength],
        "width": (1 + max_x - min_x) * cubeLength,
        "height": (1 + max_y - min_y) * cubeLength,
        "depth": (1 + max_z - min_z) * cubeLength,
    }


def detect_resolutions(dataset_path: Path, layer: str) -> Generator[Mag, None, None]:
    for mag in (dataset_path / layer).iterdir():
        try:
            yield Mag(mag.name)
        except ValueError:
            logging.info("ignoring {} as resolution".format(mag))


def detect_standard_layer(
    dataset_path: Path,
    layer_name: str,
    exact_bounding_box: Optional[dict] = None,
    category: LayerCategoryType = "color",
    layer_view_configuration: Optional[dict] = None,
) -> dict:
    # Perform metadata detection for well-known layers

    mags = list(detect_resolutions(dataset_path, layer_name))
    mags = sorted(mags)
    assert len(mags) > 0, f"No resolutions found for {dataset_path}/{layer_name}"

    if exact_bounding_box is None:
        bbox = detect_bbox(dataset_path, layer_name, mags[0])
    else:
        bbox = exact_bounding_box
    assert (
        bbox is not None
    ), f"Could not detect bounding box for {dataset_path}/{layer_name}"

    # BB can be created manually
    # assert the presence/spelling of all attributes
    assert "width" in bbox, "Attribute `width` is missing/misspelled in bounding box"
    assert "height" in bbox, "Attribute `height` is missing/misspelled in bounding box"
    assert "depth" in bbox, "Attribute `depth` is missing/misspelled in bounding box"
    assert (
        "topLeft" in bbox
    ), "Attribute `topLeft` is missing/misspelled in bounding box"

    resolutions = [
        {
            "resolution": mag.to_list(),
            "cubeLength": detect_cubeLength(dataset_path, layer_name, mag),
        }
        for mag in mags
    ]
    resolutions = [r for r in resolutions if r["cubeLength"] is not None]
    assert len(resolutions) > 0, f"No resolutions found for {dataset_path}/{layer_name}"

    dtype = detect_dtype(dataset_path, layer_name, mags[0])
    assert (
        dtype is not None
    ), f"Data type could not be detected for {dataset_path}/{layer_name}"

    layer_info = {
        "dataFormat": "wkw",
        "name": layer_name,
        "category": category,
        "elementClass": dtype,
        "boundingBox": bbox,
        "wkwResolutions": list(resolutions),
    }

    if layer_view_configuration is not None:
        layer_info["defaultViewConfiguration"] = layer_view_configuration

    return layer_info


def detect_mappings(dataset_path: Path, layer_name: str) -> List[str]:
    mapping_files = (dataset_path / layer_name / "mappings").glob("*.json")
    mapping_file_names = [mapping_file.name for mapping_file in mapping_files]
    return mapping_file_names


def detect_segmentation_layer(
    dataset_path: Path,
    layer_name: str,
    max_id: int,
    compute_max_id: bool = False,
    exact_bounding_box: Optional[dict] = None,
) -> dict:
    layer_info = detect_standard_layer(
        dataset_path, layer_name, exact_bounding_box, category="segmentation"
    )
    layer_info["mappings"] = detect_mappings(dataset_path, layer_name)
    layer_info["largestSegmentId"] = max_id

    if compute_max_id:
        logging.info("Computing max id of layer={}".format(layer_name))
        # Computing the current largest segment id
        # This may take very long due to IO load
        layer_path = str(detect_mag_path(dataset_path, layer_name, Mag(1)))
        with wkw.Dataset.open(layer_path) as dataset:
            bbox = layer_info["boundingBox"]
            layer_info["largestSegmentId"] = int(
                np.max(
                    dataset.read(
                        bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]]
                    )
                )
            )
            logging.info(
                "Max id of layer={} is {}".format(
                    layer_name, layer_info["largestSegmentId"]
                )
            )

    return layer_info


def detect_layers(
    dataset_path: Path,
    max_id: int,
    compute_max_id: bool,
    exact_bounding_box: Optional[dict] = None,
    view_configuration: Optional[dict] = None,
) -> Iterable[dict]:
    if view_configuration is None:
        view_configuration = dict()
    # Detect metadata for well-known layers (i.e., color, prediction and segmentation)
    if (dataset_path / "color").exists():
        yield detect_standard_layer(
            dataset_path,
            "color",
            exact_bounding_box,
            layer_view_configuration=view_configuration.get("color"),
        )
    if (dataset_path / "segmentation").exists():
        yield detect_segmentation_layer(
            dataset_path, "segmentation", max_id, compute_max_id, exact_bounding_box
        )
    available_layer_names = set(
        [
            Path(x).parent.parent.absolute().name
            for x in dataset_path.glob("*/*/header.wkw")
        ]
    )
    for layer_name in available_layer_names:
        # color and segmentation are already checked explicitly to ensure downwards compatibility (some older datasets don't have the header.wkw file)
        if layer_name not in ["color", "segmentation"]:
            layer_info = None
            try:
                layer_info = detect_standard_layer(
                    dataset_path,
                    layer_name,
                    exact_bounding_box,
                    layer_view_configuration=view_configuration.get(layer_name),
                )
            except Exception:
                pass
            if layer_info is not None:
                yield layer_info
            else:
                logging.warning(f"{layer_name} is not a WKW layer")


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    if not args.refresh:
        assert (
            args.scale is not None
        ), "The scale has to be specified when creating metadata for a dataset."
        assert (
            args.name is not None
        ), "Please provide a name via --name to create meta data."
        write_webknossos_metadata(
            args.path, args.name, args.scale, args.max_id, args.compute_max_id
        )
    else:
        if args.name is not None:
            logging.warning(
                "The --name argument is ignored, since --refresh was provided."
            )
        refresh_metadata(args.path, args.max_id, args.compute_max_id)
