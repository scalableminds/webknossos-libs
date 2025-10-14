import inspect
from pathlib import Path

import webknossos

"""
This script generates a mapping of all classes in webknossos to their
corresponding MkDocs pages. It is used to generate the API reference.
"""

# ruff: noqa: T201

OUT_PATH = Path("src/api")


def register_submodules(submodules_classes, mod):
    # Iterate through all the members of the module
    for name, member in inspect.getmembers(mod):
        # Check if the member is a module (submodule)
        if inspect.ismodule(member) and member.__name__.startswith("webknossos"):
            submodule_name = member.__name__

            # List classes in the submodule
            classes = inspect.getmembers(member, inspect.isclass)

            # Filter classes that are defined in this specific submodule
            defined_classes = [
                cls for cls_name, cls in classes if cls.__module__ == submodule_name
            ]

            # If there are classes defined in this submodule, add them to the dictionary
            if defined_classes:
                submodules_classes[submodule_name] = defined_classes

            register_submodules(submodules_classes, member)


def print_submodule_classes(submodules_classes):
    # Print the submodules and their classes
    for submodule, classes in submodules_classes.items():
        # The file content uses a special syntax for MkDocs to render the
        # docstrings as Markdown. The syntax is:
        # ::: module.submodule.class
        # See https://mkdocstrings.github.io/python/
        classes_string = "\n".join(
            [
                f"            - {c.__name__}"
                for c in classes
                if not c.__name__.startswith("_")
            ]
        )
        file_content = f"""::: {".".join(submodule.split(".")[0:2])}
        options:
            members:\n{classes_string}\n"""

        submodule_parts = submodule.split(".")
        submodule_name = submodule_parts[-1]
        file_name = Path(f"{submodule_name}.md")
        file_path = "/".join(submodule_parts[:-1])

        out_path = OUT_PATH.joinpath(file_path)
        out_path.mkdir(exist_ok=True, parents=True)

        print(f"Writing API docs to {out_path.joinpath(file_name)}")
        out_path.joinpath(file_name).write_text(file_content)


def main():
    # Initialize a dictionary to store submodules and their classes
    submodules_classes = {}
    register_submodules(submodules_classes, webknossos)
    print_submodule_classes(submodules_classes)


if __name__ == "__main__":
    main()
