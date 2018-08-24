import json
import re
import wkw
import logging
import numpy as np

from argparse import ArgumentParser
from glob import iglob
from os import path, makedirs, listdir


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

    parser.add_argument("--name", "-n", help="Name of the dataset")

    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25)",
        default="1,1,1",
    )

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


def write_webknossos_metadata(
    dataset_path, name, scale, max_id=0, compute_max_id=False
):

    # Generate a metadata file for webKnossos
    # Currently includes no source of information for team
    datasource_properties_path = path.join(dataset_path, "datasource-properties.json")
    layers = list(detect_layers(dataset_path, max_id, compute_max_id))
    with open(datasource_properties_path, "wt") as datasource_properties_json:
        json.dump(
            {
                "id": {"name": name, "team": "<unknown>"},
                "dataLayers": layers,
                "scale": scale,
            },
            datasource_properties_json,
            indent=2,
        )


def read_metadata_for_layer(wkw_path, layer_name):
    datasource_properties = json.load(
        open(path.join(wkw_path, "datasource-properties.json"), "r")
    )
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


def detect_dtype(dataset_path, layer, mag=1):
    layer_path = path.join(dataset_path, layer, str(mag))
    if path.exists(layer_path):
        with wkw.Dataset.open(layer_path) as dataset:
            return str(np.dtype(dataset.header.voxel_type))


def detect_cubeLength(dataset_path, layer, mag=1):
    layer_path = path.join(dataset_path, layer, str(mag))
    if path.exists(layer_path):
        with wkw.Dataset.open(layer_path) as dataset:
            return dataset.header.block_len * dataset.header.file_len


def detect_bbox(dataset_path, layer, mag=1):
    # Detect the coarse bounding box of a dataset by iterating
    # over the WKW cubes
    layer_path = path.join(dataset_path, layer, str(mag))

    def list_files(layer_path):
        return iglob(path.join(layer_path, "*", "*", "*.wkw"), recursive=True)

    def parse_cube_file_name(filename):
        CUBE_REGEX = re.compile("z(\d+)/y(\d+)/x(\d+)(\.wkw)$")
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


def detect_resolutions(dataset_path, layer):
    for mag in listdir(path.join(dataset_path, layer)):
        if re.match(r"^\d+$", mag) is not None:
            yield int(mag)


def detect_standard_layer(dataset_path, layer_name):
    # Perform metadata detection for well-known layers
    bbox = detect_bbox(dataset_path, layer_name)
    dtype = detect_dtype(dataset_path, layer_name)

    mags = list(detect_resolutions(dataset_path, layer_name))
    mags.sort()
    resolutions = [
        {
            "resolution": (mag,) * 3,
            "cubeLength": detect_cubeLength(dataset_path, layer_name, mag),
        }
        for mag in mags
    ]

    return {
        "dataFormat": "wkw",
        "name": layer_name,
        "category": layer_name,
        "elementClass": dtype,
        "boundingBox": bbox,
        "wkwResolutions": list(resolutions),
    }


def detect_segmentation_layer(dataset_path, layer_name, max_id, compute_max_id=False):
    layer_info = detect_standard_layer(dataset_path, layer_name)
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


def detect_layers(dataset_path, max_id, compute_max_id):
    # Detect metadata for well-known layers, e.g. color and segmentation
    if path.exists(path.join(dataset_path, "color")):
        yield detect_standard_layer(dataset_path, "color")
    if path.exists(path.join(dataset_path, "segmentation")):
        yield detect_segmentation_layer(
            dataset_path, "segmentation", max_id, compute_max_id
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = create_parser().parse_args()
    scale = tuple(float(x) for x in args.scale.split(","))
    write_webknossos_metadata(
        args.path, args.name, scale, args.max_id, args.compute_max_id
    )
