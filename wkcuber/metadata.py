import json
import re
import wkw
import logging
import numpy as np

from argparse import ArgumentParser
from glob import iglob
from os import path, listdir
from typing import Optional
from .mag import Mag
from typing import List


def get_datasource_path(dataset_path):
    return path.join(dataset_path, "datasource-properties.json")


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)
    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25)",
        default="1,1,1",
    )

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

    return parser


def write_datasource_properties(dataset_path, datasource_properties):
    datasource_properties_path = get_datasource_path(dataset_path)
    with open(datasource_properties_path, "wt") as datasource_properties_file:
        json.dump(datasource_properties, datasource_properties_file, indent=2)


def read_datasource_properties(dataset_path):
    with open(get_datasource_path(dataset_path), "r") as datasource_properties_file:
        return json.load(datasource_properties_file)


"""
Creates a datasource-properties.json file with the specified properties
for the given dataset path. Common layers are detected automatically.
"""


def write_webknossos_metadata(
    dataset_path,
    name,
    scale,
    max_id=0,
    compute_max_id=False,
    exact_bounding_box: Optional[dict] = None,
):

    # Generate a metadata file for webKnossos
    # Currently includes no source of information for team
    layers = list(
        detect_layers(dataset_path, max_id, compute_max_id, exact_bounding_box)
    )
    write_datasource_properties(
        dataset_path,
        {
            "id": {"name": name, "team": "<unknown>"},
            "dataLayers": layers,
            "scale": scale,
        },
    )


"""
Updates the datasource-properties.json file for a given dataset.
Use this method if you added (or removed) layers and/or changed magnifications for
existing layers.

Raises an exception if the datasource-properties.json file does not exist, yet.
In this case, use write_webknossos_metadata instead.
"""


def refresh_metadata(
    wkw_path, max_id=0, compute_max_id=False, exact_bounding_box: Optional[dict] = None
):
    dataset_path = get_datasource_path(wkw_path)
    if not path.exists(dataset_path):
        raise Exception(
            "datasource-properties.json file could not be found. Please use write_webknossos_metadata to create it."
        )

    datasource_properties = read_datasource_properties(wkw_path)
    existing_layers_dict = {
        layer["name"]: layer for layer in datasource_properties["dataLayers"]
    }

    new_layers = list(
        detect_layers(wkw_path, max_id, compute_max_id, exact_bounding_box)
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


def read_metadata_for_layer(wkw_path, layer_name):
    datasource_properties = read_datasource_properties(wkw_path)

    layers = datasource_properties["dataLayers"]
    layer_info = next(layer for layer in layers if layer["name"] == layer_name)

    dtype = np.dtype(layer_info["elementClass"])
    bounding_box = layer_info["boundingBox"]
    origin = bounding_box["topLeft"]
    bounding_box = [
        bounding_box["width"],
        bounding_box["height"],
        bounding_box["depth"],
    ]

    return layer_info, dtype, bounding_box, origin


def detect_dtype(dataset_path, layer, mag: Mag = Mag(1)):
    layer_path = path.join(dataset_path, layer, str(mag))
    if path.exists(layer_path):
        with wkw.Dataset.open(layer_path) as dataset:
            voxel_type = dataset.header.voxel_type
            num_channels = dataset.header.num_channels
            voxel_size = np.dtype(voxel_type)
            if voxel_size == np.uint8 and num_channels > 1:
                return "uint" + str(8 * num_channels)
            else:
                return str(np.dtype(voxel_type))


def detect_cubeLength(dataset_path, layer, mag: Mag = Mag(1)):
    layer_path = path.join(dataset_path, layer, str(mag))
    if path.exists(layer_path):
        with wkw.Dataset.open(layer_path) as dataset:
            return dataset.header.block_len * dataset.header.file_len


def detect_bbox(dataset_path, layer, mag: Mag = Mag(1)):
    # Detect the coarse bounding box of a dataset by iterating
    # over the WKW cubes
    layer_path = path.join(dataset_path, layer, str(mag))

    def list_files(layer_path):
        return iglob(path.join(layer_path, "*", "*", "*.wkw"), recursive=True)

    def parse_cube_file_name(filename):
        CUBE_REGEX = re.compile(r"z(\d+)/y(\d+)/x(\d+)(\.wkw)$")
        m = CUBE_REGEX.search(filename)
        return (int(m.group(3)), int(m.group(2)), int(m.group(1)))

    def list_cubes(layer_path):
        return (parse_cube_file_name(f) for f in list_files(layer_path))

    xs, ys, zs = list(zip(*list_cubes(layer_path)))

    min_x, min_y, min_z = min(xs), min(ys), min(zs)
    max_x, max_y, max_z = max(xs), max(ys), max(zs)

    cubeLength = detect_cubeLength(dataset_path, layer, mag)

    return {
        "topLeft": [min_x * cubeLength, min_y * cubeLength, min_z * cubeLength],
        "width": (1 + max_x - min_x) * cubeLength,
        "height": (1 + max_y - min_y) * cubeLength,
        "depth": (1 + max_z - min_z) * cubeLength,
    }


def detect_resolutions(dataset_path, layer) -> List[Mag]:
    for mag in listdir(path.join(dataset_path, layer)):
        try:
            yield Mag(mag)
        except ValueError:
            logging.info("ignoring {} as resolution".format(mag))


def detect_standard_layer(dataset_path, layer_name, exact_bounding_box=None):
    # Perform metadata detection for well-known layers

    mags = list(detect_resolutions(dataset_path, layer_name))
    mags = sorted(mags)
    assert len(mags) > 0, "No resolutions found"

    if exact_bounding_box is None:
        bbox = detect_bbox(dataset_path, layer_name, mags[0])
    else:
        bbox = exact_bounding_box

    resolutions = [
        {
            "resolution": mag.to_array(),
            "cubeLength": detect_cubeLength(dataset_path, layer_name, mag),
        }
        for mag in mags
    ]

    dtype = detect_dtype(dataset_path, layer_name, mags[0])

    return {
        "dataFormat": "wkw",
        "name": layer_name,
        "category": layer_name,
        "elementClass": dtype,
        "boundingBox": bbox,
        "wkwResolutions": list(resolutions),
    }


def detect_segmentation_layer(
    dataset_path, layer_name, max_id, compute_max_id=False, exact_bounding_box=None
):
    layer_info = detect_standard_layer(dataset_path, layer_name, exact_bounding_box)
    layer_info["mappings"] = []
    layer_info["largestSegmentId"] = max_id

    if compute_max_id:
        logging.info("Computing max id of layer={}".format(layer_name))
        # Computing the current largest segment id
        # This may take very long due to IO load
        layer_path = path.join(dataset_path, layer_name, "1")
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


def detect_layers(dataset_path, max_id, compute_max_id, exact_bounding_box=None):
    # Detect metadata for well-known layers (i.e., color, prediction and segmentation)
    for layer_name in ["color", "prediction"]:
        if path.exists(path.join(dataset_path, layer_name)):
            yield detect_standard_layer(dataset_path, layer_name, exact_bounding_box)
    if path.exists(path.join(dataset_path, "segmentation")):
        yield detect_segmentation_layer(
            dataset_path, "segmentation", max_id, compute_max_id, exact_bounding_box
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = create_parser().parse_args()
    if not args.refresh:
        assert (
            args.name is not None
        ), "Please provide a name via --name to create meta data."
        scale = tuple(float(x) for x in args.scale.split(","))
        write_webknossos_metadata(
            args.path, args.name, scale, args.max_id, args.compute_max_id
        )
    else:
        if args.name is not None:
            logging.warn(
                "The --name argument is ignored, since --refresh was provided."
            )
        refresh_metadata(args.path, args.max_id, args.compute_max_id)
