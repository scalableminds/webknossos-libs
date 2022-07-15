# Change Log

All notable changes to the cluser_tools library are documented in this file.

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


## [0.10.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.6) - 2022-06-27
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.5...v0.10.6)


## [0.10.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.5) - 2022-06-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.4...v0.10.5)


## [0.10.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.4) - 2022-06-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.3...v0.10.4)


## [0.10.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.3) - 2022-06-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.2...v0.10.3)

### Added
- Detect when slurm jobs crash due to being out-of-memory. [#739](https://github.com/scalableminds/webknossos-libs/pull/739)


## [0.10.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.2) - 2022-05-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.1...v0.10.2)

### Fixed
- Fix `_log() got unexpected keyword argument: 'file'` error for newer Python versions. [#735](https://github.com/scalableminds/webknossos-libs/pull/735)


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


## [0.9.21](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.21) - 2022-04-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.20...v0.9.21)


## [0.9.20](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.20) - 2022-04-20
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.19...v0.9.20)


## [0.9.19](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.19) - 2022-04-11
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.18...v0.9.19)

### Fixed
- Fixed that the ProcessPoolExecutor by the cluster tools would also create a checkpoint if the job failed. This was a regression introduced by #686. [#692](https://github.com/scalableminds/webknossos-libs/pull/692)


## [0.9.18](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.18) - 2022-04-06
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.17...v0.9.18)

### Breaking Changes
- The cluster-tools serialize the output of a job in the format `(wasSuccessful, result_value)` to a pickle file if `output_pickle_path` is provided and multiprocessing is used. This is consistent with how it is already done when using a cluster executor (e.g., slurm). [#686](https://github.com/scalableminds/webknossos-libs/pull/686)


## [0.9.17](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.17) - 2022-04-05
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.16...v0.9.17)


## [0.9.16](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.16) - 2022-03-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.15...v0.9.16)


## [0.9.15](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.15) - 2022-03-31
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.14...v0.9.15)

### Breaking Changes
- The `multiprocessing` executor now uses `spawn` as default start method. `fork` and `forkserver` can be used by supplying a `start_method` argument (e.g. `cluster_tools.get_executor("multiprocessing", start_method="forkserver")`) or by setting the `MULTIPROCESSING_DEFAULT_START_METHOD` environment variable. [#662](https://github.com/scalableminds/webknossos-libs/pull/662)


## [0.9.14](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.14) - 2022-03-24
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.13...v0.9.14)


## [0.9.13](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.13) - 2022-03-24
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.12...v0.9.13)


## [0.9.12](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.12) - 2022-03-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.11...v0.9.12)


## [0.9.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.11) - 2022-03-16
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.10...v0.9.11)


## [0.9.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.10) - 2022-03-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.9...v0.9.10)


## [0.9.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.9) - 2022-03-03
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.8...v0.9.9)


## [0.9.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.8) - 2022-02-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.7...v0.9.8)


## [0.9.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.7) - 2022-02-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.6...v0.9.7)

### Added
* Added `KubernetesExecutor` for parallelizing Python scripts on a Kubernetes cluster. [#600](https://github.com/scalableminds/webknossos-libs/pull/600)


## [0.9.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.6) - 2022-02-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.5...v0.9.6)

### Added
- Automatically detect when a multiprocessing context is set up without using an `if __name__ == "__main__"` guard in the main module. [#598](https://github.com/scalableminds/webknossos-libs/pull/598)


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
