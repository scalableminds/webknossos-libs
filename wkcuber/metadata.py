import json
from os import path, makedirs


def write_webknossos_metadata(dataset_base_path,
                              name,
                              scale,
                              dtype,
                              layer_name,
                              layer_type,
                              bbox,
                              resolutions):

    if not path.exists(dataset_base_path):
        makedirs(dataset_base_path)
        
    # Generate a combined metadata file (for webknossos). Currently have no source of information for team
    datasource_properites_path = path.join(dataset_base_path, 'datasource-properties.json')
    with open(datasource_properties_path, 'wt') as datasource_properties_json:
      json.dump({
            'id': {
                'name': name,
                'team': '',
            },
            'dataLayers': [
                {
                    'dataFormat': 'knossos',
                    'name': layer_name,
                    'category': layer_type,
                    'elementClass': dtype,
                    'sections': [
                        {
                            'name': '',
                            'resolutions': resolutions,
                            'boundingBox': {
                                "topLeft": [0, 0, 0],
                                "width": bbox[0],
                                "height": bbox[1],
                                "depth": bbox[2]
                            }
                        }
                    ]
                }
            ],
            'scale': scale
      }, datasource_properties_json)

    # These can possibly be removed but should be deprecated for a period of time
    settings_json_path = path.join(dataset_base_path, 'settings.json')
    with open(settings_json_path, 'wt') as settings_json:
        json.dump({
            'name': name,
            'priority': 0,
            'scale': scale
        }, settings_json)

    if not path.exists(path.join(dataset_base_path, layer_name)):
        makedirs(path.join(dataset_base_path, layer_name))

    layer_json_path = path.join(dataset_base_path, layer_name, 'layer.json')
    with open(layer_json_path, 'wt') as layer_json:
        json.dump({
            'typ': layer_type,
            'class': dtype
        }, layer_json)

    section_json_path = path.join(
        dataset_base_path, layer_name, 'section.json')
    with open(section_json_path, 'wt') as section_json:
        json.dump({
            'bbox': (
                (0, bbox[0]),
                (0, bbox[1]),
                (0, bbox[2])
            ),
            'resolutions': resolutions
        }, section_json)
