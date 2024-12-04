import inspect
from logging import getLogger
from pathlib import Path

import webknossos

"""
This script generates a mapping of all classes in webknossos to their
corresponding MkDocs pages. It is used to generate the API reference.
"""

logger = getLogger(__name__)

OUT_PATH = Path("src/api")

# Initialize a dictionary to store submodules and their classes
submodules_classes = {}

# Iterate through all the members of the module
for name, member in inspect.getmembers(webknossos):
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
    file_content = f"""::: {submodule}
    options:
        members:\n{classes_string}\n"""

    submodule_parts = submodule.split(".")
    submodule_name = submodule_parts[-1]
    file_name = Path(f"{submodule_name}.md")
    file_path = "/".join(submodule_parts[:-1])

    out_path = OUT_PATH.joinpath(file_path)
    out_path.mkdir(exist_ok=True, parents=True)

    logger.debug(f"Writing API docs to {out_path.joinpath(file_name)}")
    out_path.joinpath(file_name).write_text(file_content)
