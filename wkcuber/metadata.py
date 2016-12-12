import json
from os import path, makedirs


def write_webknossos_metadata(dataset_base_path,
                              name,
                              scale,
                              dtype,
                              bbox,
                              resolutions):

    if not path.exists(dataset_base_path):
        makedirs(dataset_base_path)

    settings_json_path = path.join(dataset_base_path, 'settings.json')
    with open(settings_json_path, 'wt') as settings_json:
        json.dump({
            'name': name,
            'priority': 0,
            'scale': scale
        }, settings_json)

    if not path.exists(path.join(dataset_base_path, 'color')):
        makedirs(path.join(dataset_base_path, 'color'))

    layer_json_path = path.join(dataset_base_path, 'color', 'layer.json')
    with open(layer_json_path, 'wt') as layer_json:
        json.dump({
            'typ': 'color',
            'class': dtype
        }, layer_json)

    section_json_path = path.join(dataset_base_path, 'color', 'section.json')
    with open(section_json_path, 'wt') as section_json:
        json.dump({
            'bbox': (
                (0, bbox[0]),
                (0, bbox[1]),
                (0, bbox[2])
            ),
            'resolutions': resolutions
        }, section_json)
