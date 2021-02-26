from argparse import ArgumentParser, Namespace
from os import path, sep
from pathlib import Path
from typing import Iterable, List, Any, Tuple, cast, Dict

from natsort import natsorted

from .__main__ import (
    main as convert_image_stack,
    create_parser as create_image_stack_parser,
)
from .convert_knossos import (
    main as convert_knossos,
    create_parser as create_knossos_parser,
)
from .convert_nifti import main as convert_nifti
from .image_readers import image_reader
from .utils import find_files, add_scale_flag, logger


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Input file or directory containing the input files."
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset."
    )

    add_scale_flag(parser)

    return parser


def put_default_if_not_present(args: Namespace, name: str, default: Any) -> None:
    dictionary = vars(args)
    if name not in dictionary:
        dictionary[name] = default


def put_default_from_argparser_if_not_present(
        args: Namespace, argparser: ArgumentParser, name: str
) -> None:
    put_default_if_not_present(args, name, argparser.get_default(name))


def add_to_set_in_dictionary(dictionary: dict, key: str, value: Any) -> None:
    dictionary.setdefault(key, set())
    dictionary[key].add(value)


def get_source_files(
        input_path: str, extensions: Iterable[str], allows_single_file_input: bool
) -> List[str]:
    if Path(input_path).is_dir():
        input_path = path.join(input_path, "**")
    elif not allows_single_file_input:
        return []

    source_files = list(find_files(input_path, extensions))

    return source_files


class Converter:
    def accepts_input(self, source_path: str) -> bool:
        pass

    def convert_input(self, args: Namespace) -> None:
        pass


class WkwConverter(Converter):
    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".wkw"}, False)
        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        logger.info("Already a WKW dataset. No conversion necessary...")


class NiftiConverter(Converter):
    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".nii"}, True)
        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        logger.info("Converting Nifti dataset")
        convert_nifti(args)


class KnossosConverter(Converter):
    def __init__(self) -> None:
        self.source_files: List[str] = []

    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".raw"}, False)
        self.source_files = source_files

        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        logger.info("Converting KNOSSOS dataset")

        # add missing config attributes with defaults
        knossos_parser = create_knossos_parser()
        put_default_from_argparser_if_not_present(args, knossos_parser, "mag")
        put_default_from_argparser_if_not_present(args, knossos_parser, "verbose")
        put_default_from_argparser_if_not_present(args, knossos_parser, "jobs")
        put_default_from_argparser_if_not_present(
            args, knossos_parser, "distribution_strategy"
        )

        if not hasattr(args, "dtype"):
            logger.info("Assumed data type is uint8")

        put_default_from_argparser_if_not_present(args, knossos_parser, "dtype")

        (
            dataset_name,
            layer_paths_and_mag,
        ) = self.detect_dataset_and_layer_paths_with_mag()
        args.name = dataset_name if dataset_name != ".." else "dataset"

        for layer_path, mag in layer_paths_and_mag.items():
            args.source_path = path.join(layer_path, mag)
            args.layer_name = "color" if layer_path == "" or layer_path == ".." else path.basename(layer_path)

            convert_knossos(args)

    def detect_dataset_and_layer_paths_with_mag(self) -> Tuple[str, Dict[str, str]]:
        # Path structure for knossos is .../(dataset_name)/(layer_name)/(mag)folder/x0000/y0000/z0000/filename.raw
        layer_path_to_mag_set: Dict[str, set] = dict()
        dataset_names = set()

        first_split_path = path.normpath(self.source_files[0]).split(sep)
        traversal_depth = len(first_split_path)
        starts_with_root = first_split_path[0] == ""

        assert (
            traversal_depth >= 4 if not starts_with_root else traversal_depth >= 5
        ), "Input Format is unreadable. Make sure to pass the path which points at least to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
        assert all(map(lambda p: len(path.normpath(p).split(sep)) == traversal_depth, self.source_files))

        handle_function = self.handle_path_length_longer
        if traversal_depth == 4:
            handle_function = self.handle_path_length_4
        elif traversal_depth == 5:
            handle_function = self.handle_path_length_5
        elif traversal_depth == 6:
            handle_function = self.handle_path_length_6

        for f in self.source_files:
            split_path = path.normpath(f).split(sep)
            if starts_with_root:
                split_path = split_path[1:]

            handle_function(
                split_path, layer_path_to_mag_set, dataset_names, starts_with_root
            )

        assert (
                len(dataset_names) == 1
        ), "More than one dataset found. Stopping conversion..."
        assert len(layer_path_to_mag_set) > 0, "No layers found. Stopping conversion..."

        layer_path_to_mag = dict()

        for layer, mags in layer_path_to_mag_set.items():
            layer_path_to_mag[layer] = natsorted(list(mags))[0]

        return dataset_names.pop(), layer_path_to_mag

    def handle_path_length_4(
            self,
            split_path: List[str],  # pylint: disable=unused-argument
            layer_path_to_mag_set: Dict[str, set],
            dataset_names: set,
            starts_with_root: bool,
    ):
        # already inside the mag folder => (/)x0000/y0000/z0000/filename.raw
        layer_path = self.get_layer_name_with_root_if_needed("", starts_with_root)
        add_to_set_in_dictionary(layer_path_to_mag_set, layer_path, "")
        dataset_names.add("dataset")

    def handle_path_length_5(
            self,
            split_path: List[str],
            layer_path_to_mag_set: Dict[str, set],
            dataset_names: set,
            starts_with_root: bool,
    ):
        # only the mag folder is given, therefore the layer path is empty => (/)mag/x0000/y0000/z0000/filename.raw
        layer_path = self.get_layer_name_with_root_if_needed("", starts_with_root)
        add_to_set_in_dictionary(layer_path_to_mag_set, layer_path, split_path[0])
        dataset_names.add("dataset")

    def handle_path_length_6(
            self,
            split_path: List[str],
            layer_path_to_mag_set: Dict[str, set],
            dataset_names: set,
            starts_with_root: bool,
    ):
        # additionally the layer folder is given, that should indicate a single layer as well => (/)layer/mag/x0000/y0000/z0000/filename.raw
        layer_path = split_path[0]
        layer_path = self.get_layer_name_with_root_if_needed(
            layer_path, starts_with_root
        )
        add_to_set_in_dictionary(layer_path_to_mag_set, layer_path, split_path[1])
        dataset_names.add("dataset")

    def handle_path_length_longer(
            self,
            split_path: List[str],
            layer_path_to_mag_set: Dict[str, set],
            dataset_names: set,
            starts_with_root: bool,
    ):
        # additionally the layer folder is given, that should indicate a single layer as well => (/../)dataset_name/layer/mag/x0000/y0000/z0000/filename.raw
        # also a dataset folder is given
        layer_path = sep.join(split_path[0:-5])
        layer_path = self.get_layer_name_with_root_if_needed(
            layer_path, starts_with_root
        )
        add_to_set_in_dictionary(layer_path_to_mag_set, layer_path, split_path[-5])
        dataset_names.add(split_path[-7])

    def get_layer_name_with_root_if_needed(
            self, layer_path: str, starts_with_root: bool
    ) -> str:
        if starts_with_root:
            layer_path = "/" + layer_path
        return layer_path


class ImageStackConverter(Converter):
    def __init__(self) -> None:
        self.source_files: List[str] = []
        self.args: Namespace = Namespace()

    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, image_reader.readers.keys(), True)

        if len(source_files) == 0:
            return False

        _, ext = path.splitext(source_files[0])

        assert all(
            map(lambda p: path.splitext(p)[1] == ext, source_files)
        ), "Not all image files are of the same type"

        self.source_files = source_files

        return True

    def convert_input(self, args: Namespace) -> None:
        logger.info("Converting image stack")

        # add missing config attributes with defaults
        image_stack_parser = create_image_stack_parser()
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "target_mag"
        )
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "wkw_file_len"
        )
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "interpolation_mode"
        )
        put_default_from_argparser_if_not_present(args, image_stack_parser, "start_z")
        put_default_from_argparser_if_not_present(args, image_stack_parser, "jobs")
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "distribution_strategy"
        )
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "job_resources"
        )
        put_default_from_argparser_if_not_present(args, image_stack_parser, "pad")
        put_default_from_argparser_if_not_present(args, image_stack_parser, "max_mag")
        put_default_from_argparser_if_not_present(
            args, image_stack_parser, "no_compress"
        )
        put_default_from_argparser_if_not_present(args, image_stack_parser, "isotropic")
        put_default_from_argparser_if_not_present(args, image_stack_parser, "verbose")

        # detect layer and ds name
        dataset_name, layer_path_to_name = self.detect_dataset_name_and_layer_path_to_layer_name()
        put_default_if_not_present(args, "name", dataset_name)

        for layer_path, layer_name in layer_path_to_name.items():
            args.layer_name = layer_name
            args.source_path = layer_path
            convert_image_stack(args)

    def detect_dataset_name_and_layer_path_to_layer_name(self) -> Tuple[str, Dict[str, str]]:
        parent_paths_of_images = set()
        for f in self.source_files:
            p = path.dirname(f)
            if p != "":  # p can only be empty for a single file inside the current working directory
                parent_paths_of_images.add(p)

        if len(parent_paths_of_images) == 0:
            # if no parent folder is found, it is implicitly clear that there is only one source file
            return path.splitext(path.basename(self.source_files[0]))[0], {"": "color"}

        one_path = parent_paths_of_images.pop()
        dataset_path = path.dirname(one_path)
        assert all(
            map(lambda p: path.dirname(p) == dataset_path, parent_paths_of_images)
        ), "Not all image files can be mapped to a sound layer structure"

        if dataset_path == "":
            # this means that all source files are from one folder
            dataset_name = path.basename(one_path)
            if dataset_name in ["color", "segmentation", "mask"]:
                return "dataset", {one_path: dataset_name}
            elif len(self.source_files) == 1:
                return dataset_name, {
                    one_path: path.splitext(path.basename(self.source_files[0]))[0]
                }
            else:
                return dataset_name, {one_path: "color"}
        else:
            dataset_name = path.basename(dataset_path)
            parent_paths_of_images.add(one_path)
            layer_path_to_layer_name: Dict[str, str] = dict()
            for p in parent_paths_of_images:
                layer_path_to_layer_name[p] = path.basename(p)

            return dataset_name, layer_path_to_layer_name


class ConverterManager:
    def __init__(self) -> None:
        self.converter: List[Converter] = [
            WkwConverter(),
            NiftiConverter(),
            KnossosConverter(),
            ImageStackConverter(),
        ]


def main(args: Namespace):
    matching_converters = list(
        filter(
            lambda c: c.accepts_input(args.source_path),
            converter_manager.converter,
        )
    )

    if len(matching_converters) == 0:
        logger.info(
            "No converter found. Please check the source path."
        )
        exit(1)
    elif len(matching_converters) > 1:
        logger.info(
            "Multiple converters found. Check if your source path contains multiple datasets."
        )
        exit(1)

    matching_converters[0].convert_input(args)


if __name__ == "__main__":
    parsed_args = create_parser().parse_args()
    converter_manager = ConverterManager()

    main(parsed_args)
