import shutil
from logging import getLogger
from pathlib import Path

import webknossos

'''
This script generates a mapping of all classes in webknossos to their
corresponding MkDocs pages. It is used to generate the API reference.
'''

logger = getLogger(__name__)


OUT_PATH = Path("src/api")

if OUT_PATH.joinpath("webknossos").exists():
    shutil.rmtree(OUT_PATH.joinpath("webknossos"))

# key + ":" + value.__module__
for key, value in webknossos.__dict__.items():
    if getattr(value, "__module__", "").startswith("webknossos"):

        logger.debug("Processing module", key)
        
        module = value.__module__
        
        module_parts = module.split('.')
        module_path = "/".join(module_parts[:-1])
        file_name = f"{key.lower()}.md"
        
        # The file content uses a special syntax for MkDocs to render the
        # docstrings as Markdown. The syntax is:
        # ::: module.submodule.class
        # See https://mkdocstrings.github.io/python/
        file_content = f"""::: {module}\n\n"""
        
        out_path=OUT_PATH.joinpath(module_path)
        out_path.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Writing API docs to{out_path.joinpath(file_name)}")
        # Write to file in append mode to merge several classes of a single (sub)module into one file
        with out_path.joinpath(file_name).open("a") as f:
            f.write(file_content)
