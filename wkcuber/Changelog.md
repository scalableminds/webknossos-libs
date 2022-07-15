# Changelog

All notable changes to webknossos-cuber are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/) `MAJOR.MINOR.PATCH`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.8...HEAD)

### Breaking Changes

### Added

### Changed

### Fixed


## [0.10.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.8) - 2022-07-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.7...v0.10.8)


## [0.10.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.7) - 2022-07-14
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.6...v0.10.7)

### Changed
- Made the dataset upload more robust against network errors. [#757](https://github.com/scalableminds/webknossos-libs/pull/757)


## [0.10.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.6) - 2022-06-27
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.5...v0.10.6)

### Changed
- Make KNOSSOS conversion compatible with mag-prefixed mag folders. [#756](https://github.com/scalableminds/webknossos-libs/pull/756)
- When using multiprocessing, warning filters are set up to behave as in the
  spawning context. [#741](https://github.com/scalableminds/webknossos-libs/pull/741)

### Fixed
- Fixed broken KNOSSOS to wkw conversion. [#756](https://github.com/scalableminds/webknossos-libs/pull/756)


## [0.10.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.5) - 2022-06-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.4...v0.10.5)


## [0.10.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.4) - 2022-06-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.3...v0.10.4)


## [0.10.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.3) - 2022-06-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.2...v0.10.3)

### Fixed
- Fixed a bug where nifti datasets would not be converted if called from the CLI. [#733](https://github.com/scalableminds/webknossos-libs/pull/733)



## [0.10.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.2) - 2022-05-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.1...v0.10.2)

### Changed
- Added Python 3.9 support to wkcuber [#716](https://github.com/scalableminds/webknossos-libs/pull/716)


## [0.10.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.1) - 2022-05-10
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.0...v0.10.1)


## [0.10.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.0) - 2022-05-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.24...v0.10.0)


## [0.9.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.24) - 2022-05-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.23...v0.9.24)


## [0.9.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.23) - 2022-05-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.22...v0.9.23)


## [0.9.22](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.22) - 2022-05-02
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.21...v0.9.22)

### Breaking Changes
- Deprecated `--scale` in favor of `--voxel_size`. [#704](https://github.com/scalableminds/webknossos-libs/pull/704)

### Fixed
 - Fixed a bug where upper-case file extensions would lead to errors during channel count detection. [#709](https://github.com/scalableminds/webknossos-libs/pull/709)



## [0.9.21](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.21) - 2022-04-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.20...v0.9.21)


## [0.9.20](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.20) - 2022-04-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.19...v0.9.20)

### Breaking Changes
- Deprecated `--wkw_file_len` flags in favor of `--chunks_per_shard`. [#679](https://github.com/scalableminds/webknossos-libs/pull/679)
- Deprecated `wkcuber.api` in favor of the high-level `webknososs.Dataset` API. [#679](https://github.com/scalableminds/webknossos-libs/pull/679)
- Deprecated external use of `wkcuber.utils`. [#679](https://github.com/scalableminds/webknossos-libs/pull/679)
- Remove the deprecated modules `wkcuber.downsampling_utils` and `wkcuber.upsampling_utils`. Use the high-level `Layer.downsample` and `Layer.upsample` methods in the `webknossos` package instead. [#679](https://github.com/scalableminds/webknossos-libs/pull/679)

### Added
- Added Zarr support for `wkcuber`, `wkcuber.cubing`, `wkcuber.converter`, `wkcuber.convert_knossos`, `wkcuber.convert_image_stack_to_wkw`, `wkcuber.convert_nifti`, `wkcuber.convert_raw`, `wkcuber.convert_zarr`, and `wkcuber.recubing`. These commands now take a `--data_format` flag that can either be `wkw` or `zarr`. Additionally, `--chunk_size` and `--chunks_per_shard` flag are available and take either a single number or a 3-tuple (e.g. `32,32,32`). [#689](https://github.com/scalableminds/webknossos-libs/pull/679)


## [0.9.19](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.19) - 2022-04-11
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.18...v0.9.19)

### Added
- Added support for converting `.bmp` image files. [#689](https://github.com/scalableminds/webknossos-libs/pull/689)


## [0.9.18](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.18) - 2022-04-06
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.17...v0.9.18)


## [0.9.17](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.17) - 2022-04-05
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.16...v0.9.17)


## [0.9.16](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.16) - 2022-03-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.15...v0.9.16)


## [0.9.15](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.15) - 2022-03-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.14...v0.9.15)


## [0.9.14](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.14) - 2022-03-24
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.13...v0.9.14)


## [0.9.13](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.13) - 2022-03-24
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.12...v0.9.13)

### Fixed
- Fixed that wkcuber.downsampling didn't support anisotropic downsampling for some downsampling modes like `nearest`. [#643](https://github.com/scalableminds/webknossos-libs/pull/643)


## [0.9.12](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.12) - 2022-03-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.11...v0.9.12)

### Added
- Added logging to file in `./logs` directory. [#641](https://github.com/scalableminds/webknossos-libs/pull/641)


## [0.9.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.11) - 2022-03-16
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.10...v0.9.11)


## [0.9.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.10) - 2022-03-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.9...v0.9.10)

### Fixed
- Fixed that wkcuber.downsampling didn't support an anisotropic mag for `--from`. [#638](https://github.com/scalableminds/webknossos-libs/pull/638)
- Fixed that wkcuber.downsampling didn't provide a meaningful error message when trying to do downsampling with an unsupported interpolation mode. [#619](https://github.com/scalableminds/webknossos-libs/pull/619)

## [0.9.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.9) - 2022-03-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.8...v0.9.9)


## [0.9.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.8) - 2022-02-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.7...v0.9.8)


## [0.9.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.7) - 2022-02-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.6...v0.9.7)


## [0.9.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.6) - 2022-02-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.5...v0.9.6)


## [0.9.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.5) - 2022-02-10
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.4...v0.9.5)


## [0.9.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.4) - 2022-02-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.3...v0.9.4)


## [0.9.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.3) - 2022-02-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.2...v0.9.3)


## [0.9.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.2) - 2022-02-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.1...v0.9.2)


## [0.9.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.1) - 2022-01-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.0...v0.9.1)

### Added
- Added `wkcuber.convert_zarr` tool to convert zarr files to wkw datasets. [#549](https://github.com/scalableminds/webknossos-libs/pull/549)

### Fixed
- Fixed automatic conversion of 3D tiff files which only have a single page. [#575](https://github.com/scalableminds/webknossos-libs/pull/575)


## [0.9.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.0) - 2022-01-19
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.31...v0.9.0)

### Added
- `wkcuber.upload` for uploading local datasets to webKnossos. [#544](https://github.com/scalableminds/webknossos-libs/pull/544)

### Changed
- Logging is now set to log level `INFO` by default. `DEBUG` logging can be enabled with the `--verbose` flag. Consequently, the `--silent` flag has been removed. [#544](https://github.com/scalableminds/webknossos-libs/pull/544)


## [0.8.31](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.31) - 2022-01-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.30...v0.8.31)


## [0.8.30](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.30) - 2021-12-27
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.29...v0.8.30)


## [0.8.29](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.29) - 2021-12-14
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.28...v0.8.29)


## [0.8.28](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.28) - 2021-12-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.27...v0.8.28)

### Changed
- Improved the performance of cubing and tile-cubing and integrated the dataset API into tile-cubing. [#480](https://github.com/scalableminds/webknossos-libs/pull/480)


## [0.8.27](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.27) - 2021-12-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.25...v0.8.27)

### Added
- Added importable `cube_with_args` function to main module of wkcuber. [#507](https://github.com/scalableminds/webknossos-libs/pull/507)


## [v0.8.25](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.25) - 2021-12-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.24...v0.8.25)

### Added
- `wkcuber.convert_raw` conversion tool for raw binary data files. [#498](https://github.com/scalableminds/webknossos-libs/pull/498)
- Added the `wkcuber` executable that is installed when the package is installed. [#495](https://github.com/scalableminds/webknossos-libs/pull/495)

## [v0.8.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.24) - 2021-11-30
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.23...v0.8.24)

### Fixed
- Fixed `--version` CLI argument. [#493](https://github.com/scalableminds/webknossos-libs/pull/493)

## [v0.8.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.23) - 2021-11-29
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.18...v0.8.23)

### Added
- Added the flag `--version` to `wkcuber`. [#471](https://github.com/scalableminds/webknossos-libs/pull/471)


## [v0.8.20](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.20) - 2021-10-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.19...v.8.20)

## [v0.8.19](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.19) - 2021-10-21
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.18...v.8.19)

### Fixed
- Fixed two bugs in `cubing` (regarding `start_z` and `pad`). As a result, the ImageConverters do no longer cache metadata. [#460](https://github.com/scalableminds/webknossos-libs/pull/460)

## [v0.8.18](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.18) - 2021-10-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.16...v0.8.18)

### Breaking Changes in Config & CLI
- Use Dataset API inside `cubing` to automatically write metadata. Cubing does now require a scale. [#418](https://github.com/scalableminds/webknossos-libs/pull/418)
### Added
### Changed
- Updated scikit-image dependency to 0.18.3. [#435](https://github.com/scalableminds/webknossos-libs/pull/435)
- Improved the `TIFF` and `CZI` reader to work with a wider variety of formats. The module `convert_image_stack_to_wkw` is now capable of making the result webKnossos compatible. [#335](https://github.com/scalableminds/webknossos-libs/pull/335)
### Fixed


## [0.8.16](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.16) - 2021-09-01
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.13...v0.8.16)

### Breaking Changes in Config & CLI

### Added
- Add `jp2` support. [#428](https://github.com/scalableminds/webknossos-libs/pull/428)

### Changed
- Adjust downsampling scheme to always try to minimize the scaled difference between the different dimensions of a mag and renamed the sampling mode `auto` to `anisotropic`. [#391](https://github.com/scalableminds/webknossos-libs/pull/391)
- Make parameter `executor` optional for `View.for_each_chunk` and `View.for_zipped_chunks`. [#404](https://github.com/scalableminds/webknossos-libs/pull/404)
- Add option to rename foreign layer with add_{symlink,copy}_layer. [#419](https://github.com/scalableminds/webknossos-libs/pull/419)

### Fixed
- Reverted that `dataset.add_symlink_layer` and `dataset.add_copy_layer` resolved the layer path if it was a symlink. [#408](https://github.com/scalableminds/webknossos-libs/pull/408)
- Fixed the string translation for `signed int` layer. [#428](https://github.com/scalableminds/webknossos-libs/pull/428)

## [0.8.13](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.13) - 2021-09-01
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.12...v0.8.13)

### Breaking Changes in Config & CLI
- Replaced the old properties classes of the Dataset API with attr-classes.
  - The `Layer.rename()` function is now replaced with the setter of `Layer.name`.
  - The functions `Layer.get_view_configuration` and `Layer.set_view_conficuration` are replaced by the property `Layer.default_view_configuration`. (Same applies to `Dataset.get_view_configuration` and `Dataset.set_view_configuration`)
  - Moved `LayerViewConfiguration` and `DatasetViewConfiguration` into `properties.py`
  - Removed `Layer.set_bounding_box_offset` and `Layer.set_bounding_box_size`.
  - Renamed `Layer.get_bounding_box()` to the property `Layer.bounding_box`. The method `Layer.set_bounding_box` is replaced with the setter of the property `Layer.bounding_box`.

### Added
- The API documentation is now hosted on a publicwebpage. [#392](https://github.com/scalableminds/webknossos-cuber/pull/392)

### Changed
- Uses the new `webknossos` package. All classes and functions are re-exported under the same names. [#398](https://github.com/scalableminds/webknossos-cuber/pull/398)

### Fixed
- Fixes support for Windows. [#394](https://github.com/scalableminds/webknossos-cuber/pull/394)

## [0.8.12](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.12) - 2021-08-19
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.11...v0.8.12)

### Breaking Changes in Config & CLI

### Added

### Changed
- Rollback `scikit-image` version from `0.18.0` to `0.16.2` because the newer version cause problems in voxelytics. [#390](https://github.com/scalableminds/webknossos-cuber/pull/390/files)

### Fixed

## [0.8.11](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.11) - 2021-08-19
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.10...v0.8.11)

### Breaking Changes in Config & CLI

### Added
- `dataset.add_symlink_layer` and `dataset.add_copy_layer` can now handle `Layer` arguments as well. The parameter `foreign_layer_path` was renamed to `foreign_layer`. [#389](https://github.com/scalableminds/webknossos-cuber/pull/389)

### Changed

### Fixed

## [0.8.10](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.10) - 2021-08-19
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.9...v0.8.10)

### Breaking Changes in Config & CLI

### Added

### Changed
- Avoid warnings for compressed/unaligned data, if the data is directly at the border of the bounding box. [#378](https://github.com/scalableminds/webknossos-cuber/pull/378)

### Fixed

## [0.8.9](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.9) - 2021-08-12
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.8...v0.8.9)

### Breaking Changes in Config & CLI

### Added

### Changed
- Bump scipy to `1.6.0` and `scikit-image` to `0.18.0` while keeping `numpy` to under `1.20.0` [#372](https://github.com/scalableminds/webknossos-cuber/pull/372/files)

### Fixed
- Fixes a bug where modifications to an existing dataset with floats as dtype failed. [#375](https://github.com/scalableminds/webknossos-cuber/pull/375)

## [0.8.8](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.8) - 2021-08-06
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.7...v0.8.8)

### Breaking Changes in Config & CLI

### Added

### Changed
- Bump cluster-tools from 1.59 to 1.60. [#373](https://github.com/scalableminds/webknossos-cuber/pull/373)
### Fixed

## [0.8.7](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.7) - 2021-08-04
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.6...v0.8.7)

### Breaking Changes in Config & CLI

### Added

### Changed
- Bump cluster-tools from 1.58 to 1.59. [#371](https://github.com/scalableminds/webknossos-cuber/pull/371)
### Fixed

## [0.8.6](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.6) - 2021-07-29
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.5...v0.8.6)

### Breaking Changes in Config & CLI

### Added
- Implement descriptive string representations for `Dataset`, `Layer`, `MagView` and `View`. [#369](https://github.com/scalableminds/webknossos-cuber/pull/369)
- Added option to rename a layer. [#368](https://github.com/scalableminds/webknossos-cuber/pull/368)

### Changed

### Fixed

## [0.8.5](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.5) - 2021-07-29
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.4...v0.8.5)

### Breaking Changes in Config & CLI
- The parameter allow_compressed_write from View.write() is now removed. Writing to compressed magnifications is now always allowed. If the user decides to write unaligned data, a warning about a possible performance impact is displayed once. [#356](https://github.com/scalableminds/webknossos-cuber/pull/356)

### Added
- Added functions to `wkcuber.api.dataset.Dataset` and `wkcuber.api.layer.Layer` to set and get the view configuration. [#344](https://github.com/scalableminds/webknossos-cuber/pull/344)
- Added functions to add mags of a foreign dataset (`Layer.add_symlink_mag` and `Layer.add_copy_mag`) [#367](https://github.com/scalableminds/webknossos-cuber/pull/367)

### Changed

### Fixed
- Fixed a bug where Dataset.add_symlink_layer(make_relative=True) failed to look up dataset properties. [#365](https://github.com/scalableminds/webknossos-cuber/pull/365)

## [0.8.4](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.4) - 2021-07-26
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.3...v0.8.4)

### Breaking Changes in Config & CLI

### Added

### Changed
- Datasets with a missing `largestSegmentId` can now be loaded with a default of `-1`. [#362](https://github.com/scalableminds/webknossos-cuber/pull/362)

### Fixed

## [0.8.3](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.3) - 2021-07-26
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.2...v0.8.3)

### Breaking Changes in Config & CLI

### Added

### Changed
- Updated `cluster-tools` to `1.58` [#361](https://github.com/scalableminds/webknossos-cuber/pull/361)

### Fixed


## [0.8.2](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.2) - 2021-07-26
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.1...v0.8.2)

### Breaking Changes in Config & CLI

### Added
- Added option `make_relative: bool` to `wkcuber.api.dataset.Dataset.add_symlink_layer` to make the symlink relative. [#360](https://github.com/scalableminds/webknossos-cuber/pull/360)

### Changed

### Fixed

## [0.8.1](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.1) - 2021-07-22
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.0...v0.8.1)

### Breaking Changes in Config & CLI

### Added
- Added `add_copy_layer()` to `wkcuber.api.dataset.Dataset` to copy the layer of a different dataset. [#345](https://github.com/scalableminds/webknossos-cuber/pull/345)
- Added `View.read_bbox()` which takes only a single bounding box as parameter (instead of an offset and size). [#347](https://github.com/scalableminds/webknossos-cuber/pull/347)

### Changed

### Fixed

## [0.8.0](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.0) - 2021-07-16
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.7.0...v0.8.0)

### Breaking Changes in Config & CLI
- Some breaking changes in the dataset API: [#339](https://github.com/scalableminds/webknossos-cuber/pull/339)
  - The interfaces of the methods `Layer.add_mag` and `Layer.get_or_add_mag` have changed: the parameter `block_type` is now replaced with `compress`.
  - Previously `Layer.mags` was of type `Dict[str, MagView]`. This was now changed to `Dict[Mag, MagView]`.
  - Renamed `LayerTypes` to `LayerCategories`.

### Added
- Added multiple small features: [#339](https://github.com/scalableminds/webknossos-cuber/pull/339)
  - Names of datasets can now be passed optionally when creating a dataset.
  - The `Layer` does now expose the `largest_segment_id`.
  - Add methods to get category specific layers for a given dataset.

## [0.7.0](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.7.0) - 2021-07-08
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.8...v0.7.0)

### Breaking Changes in Config & CLI
- Refactoring of the dataset API: [#331](https://github.com/scalableminds/webknossos-cuber/pull/331)
    - Tiff-support is now dropped (`TiffDataset` and `TiledTiffDataset` are now removed (alongside their corresponding `Layer` and `MagDataset` classes))
    - Module names are now lowercase (previously: `wkcuber.api.Dataset`, `wkcuber.api.Layer`, `wkcuber.api.View`, `wkcuber.api.properties.DatasetProperties`, `wkcuber.api.properties.LayerProperties`, `wkcuber.api.properties.ResolutionProperties`)
    - Some classes are renamed (`WKDataset` -> `Dataset`, `WKMagDataset` -> `MagView`)
    - The "Layer types" (previously `Layer.COLOR` and `Layer.SEGMENTATION`) are now moved into their own class `wkcuber.api.layer.LayerTypes`.
    - `View` (in particular `get_view()`) is refactored to be safer (this is also a breaking change).
      - The attribute `path_to_mag_dataset` was renamed to `path_to_mag_view`
      - Changes for `View.get_view()` (these changes also apply for `MagView.get_view()` (previously `MagDataset.get_view()`)):
        - The parameter `relative_offset` was renamed to `offset`.
        - The parameter `is_bounded` was dropped (`View`s are now always bounded).
        - The order of the parameters `size` and `offset` was changed, so that `offset` is now the first parameter.
    - The shorthand `wkcuber.api.dataset.Dataset.get_view()` was removed.
    - The flag `--write_tiff` of `convert_nifti` was removed.

### Added

### Changed

### Fixed
- Use an os independent path separator for regexes. [#334](https://github.com/scalableminds/webknossos-cuber/pull/334)

## [0.6.8](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.8) - 2021-06-18
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.7...v0.6.8)

### Breaking Changes in Config & CLI

### Added
- Added `docs/api.sh` which opens a server displaying the docs. It can also be used to persist the html to `docs/api` by invoking `docs/api.sh --persist`. [#322](https://github.com/scalableminds/webknossos-cuber/pull/322)

### Changed

### Fixed


## [0.6.7](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.7) - 2021-05-28
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.6...v0.6.7)

### Breaking Changes in Config & CLI

### Added
- Added distribution flags to `wkcuber` in order to limit the parallel jobs or use a cluster scheduler. [#323](https://github.com/scalableminds/webknossos-cuber/pull/323)
- Added a function for converting `BoundingBox` to `BoundingBoxNamedTuple`. [#324](https://github.com/scalableminds/webknossos-cuber/pull/324)

### Changed

### Fixed
- Fixed a bug for writing compressed data. This previously caused an error when downsampling certain datasets. [#326](https://github.com/scalableminds/webknossos-cuber/pull/326)

## [0.6.6](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.6) - 2021-05-14
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.5...v0.6.6)

### Fixed
- After downsampling data, the bounding box gets saved correctly to the `datasource-properties.json`. [#320](https://github.com/scalableminds/webknossos-cuber/pull/320)


## [0.6.5](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.5) - 2021-05-12
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.4...v0.6.5)

### Breaking Changes in Config & CLI
- The interface for `downsampling` was completely reworked. The flags `anisotropic_target_mag` and `isotropic` are now deprecated. Use `max` and `sampling_mode` instead. [#304](https://github.com/scalableminds/webknossos-cuber/pull/304)

### Changed
- Relaxes the constraint that all input files need to have the same type when auto-converting. [#317](https://github.com/scalableminds/webknossos-cuber/pull/317)

## [0.6.4](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.4) - 2021-05-12
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.3...v0.6.4)

### Fixed
- Fixed PEP 561 compatibility for type support

## [0.6.3](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.3) - 2021-05-12
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.2...v0.6.3)

### Breaking Changes in Config & CLI
- The dataset API now interprets `float` as 32bit. `double` can be passed as a string to use a floatingpoint with 64bit as dtype (specifying the dtype explicitly, e.g. `np.float64`, still works). The `datasource-properties.json` stores 32bit floats as `"float"` and 64bit floats as `"double"`. [#316](https://github.com/scalableminds/webknossos-cuber/pull/316)

### Added
- Added `py.typed` to conform PEP 561 for type support



## [0.6.2](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.2) - 2021-05-10
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.1...v0.6.2)

### Breaking Changes in Config & CLI

### Added
- Added method to Dataset API to compress `WKMagDatasets`. [#296](https://github.com/scalableminds/webknossos-cuber/pull/296)
- This PR allows the auto-conversion to select one channel at a time for conversion to create multiple layer. Thus, more channel formats are supported. [#291](https://github.com/scalableminds/webknossos-cuber/pull/291)

### Changed
- Previously `Dataset.get_or_add_layer` did not support dtypes like `np.uint8`. [#308](https://github.com/scalableminds/webknossos-cuber/pull/308)
- In a previous PR we switched from `str` to `Path` for paths. This PR allows both for the top-level methods in the Dataset API. [#307](https://github.com/scalableminds/webknossos-cuber/pull/307)
- Bump wkw from `0.1.4` to `1.1.9`. [#309](https://github.com/scalableminds/webknossos-cuber/pull/309)
- Bump cluster-tools from 1.54 to 1.56 [#315](https://github.com/scalableminds/webknossos-cuber/pull/315)

### Fixed
- Re-export WKDataset. [#312](https://github.com/scalableminds/webknossos-cuber/pull/312)

## [0.6.1](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.6.1) - 2021-04-29
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.0...v0.6.1)

This is the latest release at the time of creating this changelog.
