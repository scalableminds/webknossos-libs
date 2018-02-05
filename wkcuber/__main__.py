import shutil
import time
import logging
import os
from argparse import ArgumentParser

from .config import read_config
from .cubing import make_mag1_cubes_from_z_stack, get_cubing_info
from .downsampling import downsample
from .metadata import write_webknossos_metadata


def webknossos_cuber(config):
    source_path = config['dataset']['source_path']
    target_path = config['dataset']['target_path']
    layer_name = config['dataset']['layer_name']

    mag1_ref_time = time.time()

    input_is_cubed = os.path.exists(
        os.path.join(source_path, layer_name, "1")
    )

    cubing_info = get_cubing_info(
        config, expect_image_files=not input_is_cubed)

    if input_is_cubed:
        logging.info("Copying mag1 data from {} to {}".format(
            source_path, target_path))
        if os.path.exists(target_path):
            logging.warning("Target path exists already.")
        else:
            shutil.copytree(source_path, target_path)
    else:
        logging.info("Creating mag1 cubes from image stack {} in {}".format(
            source_path, target_path))
        make_mag1_cubes_from_z_stack(config, cubing_info)

        logging.info("Mag 1 succesfully cubed. Took {:.3f}h".format(
            (time.time() - mag1_ref_time) / 3600))

    write_webknossos_metadata(config['dataset']['target_path'],
                              config['dataset']['name'],
                              config['dataset']['scale'],
                              config['dataset']['dtype'],
                              config['dataset']['layer_name'],
                              config['dataset']['layer_type'],
                              cubing_info.bbox,
                              cubing_info.resolutions)

    total_down_ref_time = time.time()
    curr_mag = 2

    while curr_mag <= max(cubing_info.resolutions):
        downsample(config, curr_mag // 2, curr_mag)
        logging.info("Mag {0} succesfully cubed".format(curr_mag))
        curr_mag = curr_mag * 2

    logging.info("All mags generated. Took {:.3f}h"
                 .format((time.time() - total_down_ref_time) / 3600))

    logging.info('All done.')


def create_parser():
    """Creates a parser for command-line arguments.
    The parser can read 4 options:
        Optional arguments:
            --config, -c : path to a configuration file
            --name, -n : dataset name
        Positional arguments:
            source_path : path to input files
            target_path : output path
    Args:
    Returns:
        An ArgumentParser object.
    """

    parser = ArgumentParser()

    parser.add_argument(
        'source_path',
        help="Directory containing the input images.")

    parser.add_argument(
        'target_path',
        help="Output directory for the generated dataset.")

    parser.add_argument(
        '--name', '-n',
        help="Name of the dataset. If no name is specified, "
             "the source directory name will be used.")

    parser.add_argument(
        '--loglevel', '-ll',
        help="Loglevel. DEBUG, INFO, WARNING, ERROR or CRITICAL.",
        default='INFO')

    parser.add_argument(
        '--config', '-c',
        help="A configuration file. If no file is specified, `config.yml' "
             "from knossos_cuber's installation directory is used. Note that "
             "you still have to specify the input/output directory and "
             "source format via the command line.",
        default='config.yml')

    parser.add_argument(
        '--downsample', '-d',
        dest='downsample',
        action='store_true',
        help="Downsample only")

    parser.set_defaults(downsample=False)

    return parser


def main():
    PARSER = create_parser()
    ARGS = PARSER.parse_args()

    logging.basicConfig(level=getattr(logging, ARGS.loglevel.upper(), None))

    CONFIG = read_config(ARGS.config)

    CONFIG['dataset']['source_path'] = ARGS.source_path
    CONFIG['dataset']['target_path'] = ARGS.target_path
    CONFIG['dataset']['name'] = ARGS.name

    if not ARGS.downsample:
        webknossos_cuber(CONFIG)
    else:
        cubing_info = get_cubing_info(CONFIG, expect_image_files=False)
        curr_mag = 2
        while curr_mag <= max(cubing_info.resolutions):
            downsample(CONFIG, curr_mag // 2, curr_mag)
            logging.info("Mag {0} succesfully cubed".format(curr_mag))
            curr_mag = curr_mag * 2


if __name__ == '__main__':
    main()
