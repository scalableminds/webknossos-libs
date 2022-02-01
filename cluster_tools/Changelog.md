# Change Log

All notable changes to the cluser_tools library are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/) `MAJOR.MINOR.PATCH`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.1...HEAD)

### Breaking Changes

### Added

### Changed

### Fixed


## [0.9.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.1) - 2022-01-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.0...v0.9.1)


## [0.9.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.0) - 2022-01-19
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.31...v0.9.0)

### Added
- Allow to configure the MaxArraySize and MaxSubmitJobs slurm limits via the `SLURM_MAX_ARRAY_SIZE` and `SLURM_MAX_SUBMIT_JOBS` environment variables. If the environment variables are not specified, the limits are determined automatically. [#554](https://github.com/scalableminds/webknossos-libs/pull/554)


## [0.8.31](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.31) - 2022-01-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.30...v0.8.31)


## [0.8.30](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.30) - 2021-12-27
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.29...v0.8.30)


## [0.8.29](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.29) - 2021-12-14
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.28...v0.8.29)


## [0.8.28](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.28) - 2021-12-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.27...v0.8.28)

This module did not exist in 0.8.28. See the following notes about the mono-repo migration

# Mono Repo Migration

Before 0.8.29, cluser_tools did not exist in this repository. Instead, v1.1 to v1.61 were maintained in [this repository](https://github.com/scalableminds/cluster_tools/releases).
Note that 0.8.29 is newer than 1.61.
The poor naming was a sacrifice during the mono-repo migration which enforced synchronized versioning between all packages (e.g., wkcuber and webknossos).
To avoid version conflicts in the future, the first major release for webknossos-libs will be v2.
