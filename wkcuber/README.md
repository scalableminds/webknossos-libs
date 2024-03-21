# ⚠️⚠️⚠️ wkcuber ⚠️⚠️⚠️

⚠️⚠️⚠️ The `wkcuber` package is deprecated because the CLI was integrated into the `webknossos` package. Please use the `webknossos` package instead. ⚠️⚠️⚠️

## How to migrate to the new CLI

- use `pip install webknossos[all]` instead of `pip install wkcuber`
- replace `wkcuber` with `webknossos` in your requirements file (`requirements.txt`, `pyproject.toml`, `setup.py`, ...)
- if the `wkcuber` package is used by one of your dependencies it would be great if you would take the time to report the deprecation of the `wkcuber` package in the issue tracker

Run `webknossos --help` to get an overview of the currently available commands. Run `webknossos <subcommand> --help` to learn more about specific subcommands.

- These commands changed:
  - `python -m wkcuber`, `python -m wkcuber.convert_image_stack_to_wkw` → `webknossos convert`
  - `python -m wkcuber.export_wkw_as_tiff` → `webknossos export-wkw-as-tiff`
  - `python -m wkcuber.convert_knossos` → `webknossos convert-knossos`
  - `python -m wkcuber.convert_raw` → `webknossos convert-raw`
  - `python -m wkcuber.downsampling` → `webknossos downsample`
  - `python -m wkcuber.compress` → `webknossos compress`
  - `python -m wkcuber.check_equality` → `webknossos check-equality`
- There is one new command for the CLI:
  - `webknossos upload` to upload a dataset to a WEBKNOSSOS server
- These commands have been removed:
  - `python -m wkcuber.cubing`
  - `python -m wkcuber.tile_cubing`
  - `python -m wkcuber.metadata`
  - `python -m wkcuber.recubing`

## License
AGPLv3
Copyright scalable minds
