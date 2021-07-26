# Change Log

All notable changes to webknossos-cuber are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Calendar Versioning](http://calver.org/) `0Y.0M.MICRO`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.0...HEAD)

### Breaking Changes in Config & CLI

### Added

### Changed
- Datasets with a missing `largestSegmentId` can now be loaded with a default of `-1`. [#362](https://github.com/scalableminds/webknossos-cuber/pull/362)

### Fixed

## [0.8.2](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.2) - 2021-07-26
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.1...HEAD)

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
