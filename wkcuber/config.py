import yaml

DEFAULT_CONFIG = {
    'processing': {
        'buffer_size_in_cubes': 10000,
        'buffer_size_in_cubes_downsampling': 10000,
        'num_downsampling_cores': 10,
        'num_io_threads': 10,
        'cube_edge_len': 128,
        'skip_already_cubed_layers': True,
        'skip_already_downsampled_cubes': True,
    },
    'dataset': {
        'source_path': "",
        'target_path': "",
        'name': "",
        'scale': (1, 1, 1),
        'dtype': 'uint8'
    }
}


def apply_defaults(target, defaults):
    for key, value in defaults.items():
        if isinstance(value, dict):
            if key not in target:
                target[key] = {}
            apply_defaults(target[key], value)
        else:
            if key not in target:
                target[key] = value
    return target


def read_config(config_file_path):
    with open(config_file_path, 'rt') as config_yaml:
        config = yaml.load(config_yaml)

    return validate_config(config)


def validate_config(config):

    return apply_defaults(config, DEFAULT_CONFIG)
