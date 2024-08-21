import shutil
import inspect
import sys
from logging import getLogger
from pathlib import Path

import webknossos

'''
This script generates a mapping of all classes in webknossos to their
corresponding MkDocs pages. It is used to generate the API reference.
'''

logger = getLogger(__name__)

OUT_PATH = Path("src/api")

if OUT_PATH.exists():
    shutil.rmtree(OUT_PATH)

for key, value in webknossos.__dict__.items():
    if getattr(value, "__module__", "").startswith("webknossos"):

        logger.debug("Processing module", key)
        
        module = value.__module__
        
        module_parts = module.split('.')
        module_name = module_parts[-1]
        module_path = "/".join(module_parts[:-1])
        file_name = f"{key.lower()}.md"

        # Extract all classes from the module
        classes = inspect.getmembers(sys.modules[module], inspect.isclass)

        # Only include classes that are in implemented in that module
        classes = [c for c in classes if c[1].__module__ == module]
        classes = [c for c in classes if not c[0].startswith("_")]
        
        # The file content uses a special syntax for MkDocs to render the
        # docstrings as Markdown. The syntax is:
        # ::: module.submodule.class
        # See https://mkdocstrings.github.io/python/
        classes_string = "\n".join([f"            - {c[0]}" for c in classes])
        file_content = f"""::: {module}
    options:
        members:\n{classes_string}\n"""
        
        out_path=OUT_PATH.joinpath(module_path)
        out_path.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Writing API docs to{out_path.joinpath(file_name)}")
        out_path.joinpath(file_name).write_text(file_content)
