# Change Log

All notable changes to webknossos-cuber are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Calendar Versioning](http://calver.org/) `0Y.0M.MICRO`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.6.1...HEAD)

### Breaking Changes in Config & CLI
- The interface for `downsamping` was completely reworked. The flags `anisotropic_target_mag` and `isotropic` are now deprecated. Use `max` and `sampling_mode` instead. [#304](https://github.com/scalableminds/webknossos-cuber/pull/304)

### Added

### Changed

### Fixed

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
