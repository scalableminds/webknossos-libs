from argparse import ArgumentParser, Namespace
from os import path, sep
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Dict, Set, Callable, cast

from .cubing import (
    cubing as cube_image_stack,
    create_parser as create_image_stack_parser,
)
from .convert_knossos import (
    main as convert_knossos,
    create_parser as create_knossos_parser,
)
from .convert_nifti import main as convert_nifti, create_parser as create_nifti_parser
from .image_readers import image_reader
from .utils import find_files, add_scale_flag, logger, add_verbose_flag, setup_logging
from .metadata import write_webknossos_metadata


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        epilog="If you want to pass more specific config values, please use the individual converters. See the readme for a complete overview."
    )

    parser.add_argument(
        "source_path", help="Input file or directory containing the input files."
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset."
    )

    add_scale_flag(parser)
    add_verbose_flag(parser)

    return parser


def put_default_if_not_present(args: Namespace, name: str, default: Any) -> None:
    dictionary = vars(args)
    if name not in dictionary or dictionary[name] is None:
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
        self.prefix: str = ""

    def accepts_input(self, source_path: str) -> bool:
        raise NotImplementedError()

    def convert_input(self, args: Namespace) -> bool:
        # returns True if metadata should be written after the conversion
        raise NotImplementedError()

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
        ), "Cannot detect correct layer format. Please check the input directory."

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

    def convert_input(self, args: Namespace) -> bool:
        logger.info("Already a WKW dataset. No conversion necessary...")
        return False


class NiftiConverter(Converter):
    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".nii"}, True)
        self.source_files = source_files
        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> bool:
        logger.info("Converting Nifti dataset")

        # add missing config attributes with defaults
        nifti_parser = create_nifti_parser()
        if not hasattr(args, "dtype"):
            logger.info("Assumed data type is uint8")
        put_default_from_argparser_if_not_present(args, nifti_parser, "write_tiff")
        put_default_from_argparser_if_not_present(args, nifti_parser, "dtype")
        put_default_from_argparser_if_not_present(
            args, nifti_parser, "use_orientation_header"
        )
        put_default_from_argparser_if_not_present(
            args, nifti_parser, "enforce_bounding_box"
        )
        put_default_from_argparser_if_not_present(args, nifti_parser, "flip_axes")
        put_default_from_argparser_if_not_present(args, nifti_parser, "verbose")

        if len(self.source_files) == 1:
            source_file = self.source_files[0]
            layer_name = path.splitext(path.basename(source_file))[0]
            put_default_if_not_present(
                args, "is_segmentation_layer", layer_name == "segmentation"
            )
            args.layer_name = layer_name
            args.source_path = source_file
        else:
            # We do not explicitly set the "color_file" option since we could not guess it any better than the internal algorithm
            for p in self.source_files:
                layer_name = path.splitext(path.basename(p))[0]
                if layer_name == "segmentation":
                    put_default_if_not_present(
                        args, "segmentation_file", path.relpath(p, args.source_path)
                    )
                    break

        convert_nifti(args)

        return False


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

    def convert_input(self, args: Namespace) -> bool:
        logger.info("Converting KNOSSOS dataset")

        # add missing config attributes with defaults
        knossos_parser = create_knossos_parser()
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
            layer_paths_and_mags,
        ) = self.detect_dataset_and_layer_paths_with_mag()
        put_default_if_not_present(args, "name", dataset_name)

        for layer_path, mags in layer_paths_and_mags.items():
            for mag in mags:
                # if the mag path is empty, we are already inside the mag folder, so there is only mag. We guess that this is mag 1.
                if mag != "":
                    try:
                        mag_int = int(mag)
                    except ValueError:
                        continue
                else:
                    mag_int = 1
                args.mag = mag_int
                args.source_path = path.join(layer_path, mag)
                args.layer_name = (
                    "color"
                    if path.basename(layer_path) == ""
                    else path.basename(layer_path)
                )
                convert_knossos(args)

        return True

    def detect_dataset_and_layer_paths_with_mag(
        self,
    ) -> Tuple[str, Dict[str, Set[str]]]:
        # Path structure for knossos is .../(dataset_name)/(layer_name)/(mag)folder/x0000/y0000/z0000/filename.raw
        traversal_depth = self.check_path_length_and_set_prefix()
        starts_with_prefix = self.prefix != ""

        assert (
            traversal_depth >= 4 if not starts_with_prefix else traversal_depth >= 5
        ), "Input Format is unreadable. Make sure to pass the path which points at least to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."

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

        return self.dataset_names.pop(), self.layer_path_to_mag_set

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

    def convert_input(self, args: Namespace) -> bool:
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
            cube_image_stack(
                args.source_path,
                args.target_path,
                args.layer_name,
                args.batch_size if "batch_size" in args else None,
                args,
            )

        return True

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
            self.layer_path_to_layer_name[self.prefix + split_path[0]] = "color"
        else:
            self.dataset_names.add("dataset")
            self.layer_path_to_layer_name[self.prefix] = "color"

    def handle_path_length_2(
        self,
        split_path: List[str],
    ) -> None:
        if split_path[0] in ["color", "segmentation", "mask"]:
            layer_name = split_path[0]
            self.dataset_names.add("dataset")
        else:
            self.dataset_names.add(split_path[0])
            if len(self.source_files) == 1:
                layer_name = path.splitext(split_path[1])[0]
            else:
                layer_name = "color"

        if len(self.source_files) == 1:
            self.layer_path_to_layer_name[self.source_files[0]] = layer_name
        else:
            self.layer_path_to_layer_name[self.prefix + split_path[0]] = layer_name

    def handle_path_length_longer(
        self,
        split_path: List[str],
    ) -> None:
        self.dataset_names.add(split_path[-3])
        if len(self.source_files) == 1:
            self.layer_path_to_layer_name[self.source_files[0]] = split_path[-2]
        else:
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
    converter_manager = ConverterManager()

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

    should_write_metadata = matching_converters[0].convert_input(args)
    if should_write_metadata:
        write_webknossos_metadata(args.target_path, args.name, args.scale)


if __name__ == "__main__":
    parsed_args = create_parser().parse_args()
    setup_logging(parsed_args)

    main(parsed_args)
