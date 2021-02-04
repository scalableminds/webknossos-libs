import sys
from argparse import ArgumentParser, Namespace
from os import path
from pathlib import Path
from .utils import find_files
from typing import Iterable, List
from .convert_nifti import main as convert_nifti, create_parser as create_nifti_parser
from .convert_knossos import (
    main as convert_knossos,
    create_parser as create_knossos_parser,
)
from .__main__ import (
    main as convert_image_stack,
    create_parser as create_image_stack_parser,
)
from .image_readers import image_reader


def find_first_positional_argument() -> str:
    should_continue = False
    for i, arg in enumerate(sys.argv):
        if i == 0:
            continue
        elif should_continue:
            should_continue = False
        elif arg.startswith("-"):
            should_continue = True
        else:
            return arg

    raise Exception("No input path found!")


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
    def accepts_input(self, input_path: str) -> bool:
        pass

    def get_argument_parser(self) -> ArgumentParser:
        pass

    def convert_input(self, args: Namespace) -> None:
        pass


class WkwConverter(Converter):
    def accepts_input(self, input_path: str) -> bool:
        source_files = get_source_files(input_path, {".wkw"}, False)
        return len(source_files) > 0

    def get_argument_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            "source_path", help="Input directory containing the WKW dataset."
        )
        return parser

    def convert_input(self, args: Namespace) -> None:
        print("Already a WKW dataset. No conversion necessary...")


class NiftiConverter(Converter):
    def accepts_input(self, input_path: str) -> bool:
        source_files = get_source_files(input_path, {".nii"}, True)
        return len(source_files) > 0

    def get_argument_parser(self) -> ArgumentParser:
        return create_nifti_parser()

    def convert_input(self, args: Namespace) -> None:
        print("Converting Nifti dataset")
        convert_nifti(args)


class KnossosConverter(Converter):
    def accepts_input(self, input_path: str) -> bool:
        source_files = get_source_files(input_path, {".raw"}, False)
        return len(source_files) > 0

    def get_argument_parser(self) -> ArgumentParser:
        return create_knossos_parser()

    def convert_input(self, args: Namespace) -> None:
        print("Converting KNOSSOS dataset")
        convert_knossos(args)


class ImageStackConverter(Converter):
    def accepts_input(self, input_path: str) -> bool:
        source_files = get_source_files(input_path, image_reader.readers.keys(), True)

        if len(source_files) == 0:
            return False

        _, ext = path.splitext(source_files[0])

        assert all(
            map(lambda p: path.splitext(p)[1] == ext, source_files)
        ), "Not all image files are of the same type"

        return True

    def get_argument_parser(self) -> ArgumentParser:
        return create_image_stack_parser()

    def convert_input(self, args: Namespace) -> None:
        print("Converting image stack")
        convert_image_stack(args)


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
    source_path = find_first_positional_argument()

    fitting_converters = list(
        filter(
            lambda c: c.accepts_input(source_path),
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
    argument_parser: ArgumentParser = converter.get_argument_parser()
    args, _ = argument_parser.parse_known_args()

    converter.convert_input(args)
