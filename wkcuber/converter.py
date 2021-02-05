from argparse import ArgumentParser, Namespace
from os import path, cpu_count
from pathlib import Path
from .utils import find_files, add_scale_flag
from typing import Iterable, List, Any, Tuple, cast
from .convert_nifti import main as convert_nifti
from .convert_knossos import main as convert_knossos
from .__main__ import main as convert_image_stack

from .image_readers import image_reader


def get_converter_parser() -> ArgumentParser:
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


def get_source_files(
    input_path: str, extensions: Iterable[str], allows_single_file_input: bool
) -> List[str]:
    source_path = Path(input_path)

    if source_path.is_dir():
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
    def accepts_input(self, source_path: str) -> bool:
        source_files = get_source_files(source_path, {".raw"}, False)
        return len(source_files) > 0

    def convert_input(self, args: Namespace) -> None:
        print("Converting KNOSSOS dataset")
        convert_knossos(args)


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
        dataset_name, layer_names = self.detect_dataset_and_layer_names()
        put_default_if_not_present(args, "name", dataset_name)

        self.args = args

        # we can detect the layer names correctly, but we currently do not split the source files accordingly
        args.layer_name = layer_names[0]
        print(args)
        convert_image_stack(args)

    def detect_dataset_and_layer_names(self) -> Tuple[str, List[str]]:
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
                return path.basename(self.args.target_path), [dataset_name]
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


converter_manager = ConverterManager()

if __name__ == "__main__":
    args = get_converter_parser().parse_args()

    fitting_converters = list(
        filter(
            lambda c: c.accepts_input(args.source_path),
            converter_manager.converter,
        )
    )

    if len(fitting_converters) == 0:
        print(
            "No converter found. Please specify which converter you want to use or check the source path."
        )
        exit(1)
    elif len(fitting_converters) > 1:
        print(
            "Multiple converters found. Check if your source path contains multiple datasets."
        )
        exit(1)

    converter = fitting_converters[0]
    print("Choosing the", converter.__class__.__name__)

    converter.convert_input(args)
