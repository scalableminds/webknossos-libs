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
    datasource_properties_path = path.join(dataset_base_path, 'datasource-properties.json')
    with open(datasource_properties_path, 'wt') as datasource_properties_json:
        json.dump({
            'id': {
                'name': name,
                'team': '<unknown>',
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
        }, datasource_properties_json, indent=2)

