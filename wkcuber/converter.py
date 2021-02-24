from argparse import ArgumentParser, Namespace
from os import path, cpu_count, sep
from pathlib import Path
from natsort import natsorted
from .utils import find_files, add_scale_flag, logger
from typing import Iterable, List, Any, Tuple, cast, Dict
from .convert_nifti import main as convert_nifti
from .convert_knossos import main as convert_knossos
from .__main__ import main as convert_image_stack

from .image_readers import image_reader


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


def add_or_update_if_present(dictionary: dict, name: str, value: Any) -> None:
    dictionary.setdefault(name, set())
    dictionary[name].add(value)


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
        print("Already a WKW dataset. No conversion necessary...")


class NiftiConverter(Converter):
    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".nii"}, True)
        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        print("Converting Nifti dataset")
        convert_nifti(args)


class KnossosConverter(Converter):
    def __init__(self) -> None:
        self.source_files: List[str] = []

    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".raw"}, False)
        self.source_files = source_files

        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        print("Converting KNOSSOS dataset")

        # add missing config attributes with defaults
        put_default_if_not_present(args, "mag", 1)
        put_default_if_not_present(args, "verbose", True)
        put_default_if_not_present(args, "jobs", cpu_count())
        put_default_if_not_present(args, "distribution_strategy", "multiprocessing")
        if not hasattr(args, "dtype"):
            logger.info("Assumed data type is uint8")
        put_default_if_not_present(args, "dtype", "uint8")

        (
            dataset_name,
            layer_paths_and_mag,
        ) = self.detect_dataset_and_layer_paths_with_mag()
        args.name = dataset_name

        for layer_path, mag in layer_paths_and_mag.items():
            args.source_path = path.join(layer_path, mag)
            args.layer_name = "color" if layer_path == "" else path.basename(layer_path)

            convert_knossos(args)

    def detect_dataset_and_layer_paths_with_mag(self) -> Tuple[str, Dict[str, str]]:
        # Path structure for knossos is .../(dataset_name)/(layer_name)/(mag)folder/x0000/y0000/z0000/filename.raw
        layers: Dict[str, set] = dict()
        dataset_names = set()
        for f in self.source_files:
            split_path = path.normpath(f).split(sep)
            starts_with_root = split_path[0] == ""
            if starts_with_root:
                split_path = split_path[1:]
            assert (
                len(split_path) >= 4
            ), "Input Format is unreadable. Make sure to pass the path which points at least to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."

            if len(split_path) == 4:
                # already inside the mag folder
                add_or_update_if_present(layers, "", "")
                dataset_names.add("dataset")
            elif len(split_path) == 5:
                # only the mag folder is given, therefore the layer path is empty
                add_or_update_if_present(layers, "", split_path[0])
                dataset_names.add("dataset")
            elif len(split_path) == 6:
                # additionally the layer folder is given, that should indicate a single layer as well
                add_or_update_if_present(layers, split_path[0], split_path[1])
                dataset_names.add("dataset")
            else:
                # also a dataset folder is given
                layer_path = sep.join(split_path[0:-5])
                if starts_with_root:
                    layer_path = "/" + layer_path
                add_or_update_if_present(layers, layer_path, split_path[-5])
                dataset_names.add(split_path[-7])

        assert (
            len(dataset_names) == 1
        ), "More than one dataset found. Stopping conversion..."
        assert len(layers) > 0, "No layers found. Stopping conversion..."

        layer_names_and_mags = dict()

        for layer, mags in layers.items():
            layer_names_and_mags[layer] = natsorted(list(mags))[0]

        return dataset_names.pop(), layer_names_and_mags


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
        print("Converting image stack")

        # add missing config attributes with defaults
        put_default_if_not_present(args, "target_mag", 1)
        put_default_if_not_present(args, "wkw_file_len", 32)
        put_default_if_not_present(args, "interpolation_mode", "default")
        put_default_if_not_present(args, "start_z", 0)
        put_default_if_not_present(args, "jobs", cpu_count())
        put_default_if_not_present(args, "distribution_strategy", "multiprocessing")
        put_default_if_not_present(args, "job_resources", None)
        put_default_if_not_present(args, "pad", False)
        put_default_if_not_present(args, "max_mag", 32)
        put_default_if_not_present(args, "no_compress", False)
        put_default_if_not_present(args, "isotropic", False)
        put_default_if_not_present(args, "verbose", True)

        # detect layer and ds name
        dataset_name, layer_names = self.detect_dataset_and_layer_names(args)
        put_default_if_not_present(args, "name", dataset_name)

        # TODO we can detect the layer names correctly, but we currently do not split the source files accordingly
        args.layer_name = layer_names[0]
        convert_image_stack(args)

    def detect_dataset_and_layer_names(self, args: Namespace) -> Tuple[str, List[str]]:
        paths = set()
        for f in self.source_files:
            p = path.dirname(f)
            if p != "":
                paths.add(p)

        if len(paths) == 0:
            # if no parent folder is found, it is implicitly clear that there is only one source file
            return path.splitext(path.basename(self.source_files[0]))[0], ["color"]

        one_path = paths.pop()
        dataset_name = path.dirname(one_path)
        assert all(
            map(lambda p: path.dirname(p) == dataset_name, paths)
        ), "Not all image files can be ordered into layers"
        if dataset_name == "":
            # this means that all source files are from one folder
            dataset_name = path.basename(one_path)
            if dataset_name in ["color", "segmentation", "mask"]:
                return path.basename(args.target_path), [dataset_name]
            elif len(self.source_files) == 1:
                return dataset_name, [
                    path.splitext(path.basename(self.source_files[0]))[0]
                ]
            else:
                return dataset_name, ["color"]
        else:
            dataset_name = path.basename(dataset_name)
            paths.add(one_path)
            layer_names = list(map(lambda p: cast(str, path.basename(p)), paths))
            return dataset_name, layer_names


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
        print(
            "No converter found. Please specify which converter you want to use or check the source path."
        )
        exit(1)
    elif len(matching_converters) > 1:
        print(
            "Multiple converters found. Check if your source path contains multiple datasets."
        )
        exit(1)

    converter = matching_converters[0]
    print("Choosing the", converter.__class__.__name__)

    converter.convert_input(args)


if __name__ == "__main__":
    args = create_parser().parse_args()
    converter_manager = ConverterManager()

    main(args)
