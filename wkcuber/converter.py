from argparse import ArgumentParser, Namespace
from os import path, sep
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Dict, Set, Callable, cast

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
    def __init__(self) -> None:
        self.source_files: List[str] = []
        self.prefix = ""

    def accepts_input(self, source_path: str) -> bool:
        pass

    def convert_input(self, args: Namespace) -> None:
        pass

    # returns [prefix, traversal_depth]
    def check_path_length_and_set_prefix(self) -> int:
        first_split_path = self.source_files[0].split(sep)
        traversal_depth = len(first_split_path)
        self.prefix = ""
        if first_split_path[0] == "":
            self.prefix = "/"
        elif first_split_path[0] == "..":
            self.prefix = "../"

        assert all(
            map(
                lambda p: len(p.split(sep)) == traversal_depth,
                self.source_files,
            )
        )

        return traversal_depth

    def apply_handle_function(
        self, handle_function: Callable, starts_with_prefix: bool
    ) -> None:
        for f in self.source_files:
            split_path = f.split(sep)
            if starts_with_prefix:
                split_path = split_path[1:]

            handle_function(split_path)


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
        super().__init__()
        self.layer_path_to_mag_set: Dict[str, set] = dict()
        self.dataset_names: Set[str] = set()
        self.prefix: str = ""

    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".raw"}, False)
        self.source_files = list(
            map(lambda p: cast(str, path.normpath(p)), source_files)
        )

        return len(self.source_files) > 0

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
            args.layer_name = (
                "color"
                if layer_path == "" or layer_path == ".."
                else path.basename(layer_path)
            )

            convert_knossos(args)

    def detect_dataset_and_layer_paths_with_mag(self) -> Tuple[str, Dict[str, str]]:
        # Path structure for knossos is .../(dataset_name)/(layer_name)/(mag)folder/x0000/y0000/z0000/filename.raw
        traversal_depth = self.check_path_length_and_set_prefix()
        starts_with_prefix = self.prefix != ""

        assert (
            traversal_depth >= 4 if not starts_with_prefix else traversal_depth >= 5
        ), "Input Format is unreadable. Make sure to pass the path which points at least to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
        assert all(
            map(
                lambda p: len(p.split(sep)) == traversal_depth,
                self.source_files,
            )
        )

        if starts_with_prefix:
            traversal_depth = traversal_depth - 1

        if traversal_depth == 4:
            self.apply_handle_function(self.handle_path_length_4, starts_with_prefix)
        elif traversal_depth == 5:
            self.apply_handle_function(self.handle_path_length_5, starts_with_prefix)
        elif traversal_depth == 6:
            self.apply_handle_function(self.handle_path_length_6, starts_with_prefix)
        else:
            self.apply_handle_function(
                self.handle_path_length_longer, starts_with_prefix
            )

        assert (
            len(self.dataset_names) == 1
        ), "More than one dataset found. Stopping conversion..."
        assert (
            len(self.layer_path_to_mag_set) > 0
        ), "No layers found. Stopping conversion..."

        layer_path_to_mag = dict()

        for layer, mags in self.layer_path_to_mag_set.items():
            layer_path_to_mag[layer] = natsorted(list(mags))[0]

        return self.dataset_names.pop(), layer_path_to_mag

    def handle_path_length_4(
        self,
        split_path: List[str],  # pylint: disable=unused-argument
    ) -> None:
        # already inside the mag folder => (/)x0000/y0000/z0000/filename.raw
        add_to_set_in_dictionary(self.layer_path_to_mag_set, self.prefix, "")
        self.dataset_names.add("dataset")

    def handle_path_length_5(
        self,
        split_path: List[str],
    ) -> None:
        # only the mag folder is given, therefore the layer path is empty => (/)mag/x0000/y0000/z0000/filename.raw
        add_to_set_in_dictionary(self.layer_path_to_mag_set, self.prefix, split_path[0])
        self.dataset_names.add("dataset")

    def handle_path_length_6(
        self,
        split_path: List[str],
    ) -> None:
        # additionally the layer folder is given, that should indicate a single layer as well => (/)layer/mag/x0000/y0000/z0000/filename.raw
        add_to_set_in_dictionary(
            self.layer_path_to_mag_set, self.prefix + split_path[0], split_path[1]
        )
        self.dataset_names.add("dataset")

    def handle_path_length_longer(
        self,
        split_path: List[str],
    ) -> None:
        # also a dataset folder is given => (/../)dataset_name/layer/mag/x0000/y0000/z0000/filename.raw
        layer_path = self.prefix + sep.join(split_path[0:-5])
        add_to_set_in_dictionary(self.layer_path_to_mag_set, layer_path, split_path[-5])
        self.dataset_names.add(split_path[-7])


class ImageStackConverter(Converter):
    def __init__(self) -> None:
        super().__init__()
        self.args: Namespace = Namespace()
        self.layer_path_to_layer_name: Dict[str, str] = dict()
        self.dataset_names: Set[str] = set()

    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, image_reader.readers.keys(), True)

        if len(source_files) == 0:
            return False

        _, ext = path.splitext(source_files[0])

        assert all(
            map(lambda p: path.splitext(p)[1] == ext, source_files)
        ), "Not all image files are of the same type"

        self.source_files = list(
            map(lambda p: cast(str, path.normpath(p)), source_files)
        )

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
        (
            dataset_name,
            layer_path_to_name,
        ) = self.detect_dataset_name_and_layer_path_to_layer_name()
        put_default_if_not_present(args, "name", dataset_name)

        for layer_path, layer_name in layer_path_to_name.items():
            args.layer_name = layer_name
            args.source_path = layer_path
            convert_image_stack(args)

    def detect_dataset_name_and_layer_path_to_layer_name(
        self,
    ) -> Tuple[str, Dict[str, str]]:
        # path format is (.../)(dataset_name/)(layer_name/)file_name.ending
        traversal_depth = self.check_path_length_and_set_prefix()

        starts_with_prefix = self.prefix != ""
        if starts_with_prefix:
            traversal_depth = traversal_depth - 1

        if traversal_depth == 1:
            self.apply_handle_function(self.handle_path_length_1, starts_with_prefix)
        elif traversal_depth == 2:
            self.apply_handle_function(self.handle_path_length_2, starts_with_prefix)
        else:
            self.apply_handle_function(
                self.handle_path_length_longer, starts_with_prefix
            )

        assert (
            len(self.dataset_names) == 1
        ), "More than one dataset found. Stopping conversion..."
        assert (
            len(self.layer_path_to_layer_name) > 0
        ), "No layers found. Stopping conversion..."

        return self.dataset_names.pop(), self.layer_path_to_layer_name

    def handle_path_length_1(
        self,
        split_path: List[str],
    ) -> None:
        if len(self.source_files) == 1:
            self.dataset_names.add(path.splitext(split_path[0])[0])
        else:
            self.dataset_names.add("dataset")

        self.layer_path_to_layer_name[self.prefix] = "color"

    def handle_path_length_2(
        self,
        split_path: List[str],
    ) -> None:
        if split_path[0] in ["color", "segmentation", "mask"]:
            self.layer_path_to_layer_name[self.prefix + split_path[0]] = split_path[0]
            self.dataset_names.add("dataset")
        else:
            if len(self.source_files) == 1:
                self.layer_path_to_layer_name[
                    self.prefix + split_path[0]
                ] = path.splitext(split_path[1])[0]
            else:
                self.layer_path_to_layer_name[self.prefix + split_path[0]] = "color"
            self.dataset_names.add(split_path[0])

    def handle_path_length_longer(
        self,
        split_path: List[str],
    ) -> None:
        self.dataset_names.add(split_path[-3])
        self.layer_path_to_layer_name[
            self.prefix + sep.join(split_path[0:-1])
        ] = split_path[-2]


class ConverterManager:
    def __init__(self) -> None:
        self.converter: List[Converter] = [
            WkwConverter(),
            NiftiConverter(),
            KnossosConverter(),
            ImageStackConverter(),
        ]


def main(args: Namespace) -> None:
    matching_converters = list(
        filter(
            lambda c: c.accepts_input(args.source_path),
            converter_manager.converter,
        )
    )

    if len(matching_converters) == 0:
        logger.info("No converter found. Please check the source path.")
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
