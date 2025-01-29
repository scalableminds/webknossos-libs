# Changelog

All notable changes to the webknossos python library are documented in this file.

Please see the [Stability Policy](./stability_policy.md) for details about the version schema
and compatibility implications.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/) `MAJOR.MINOR.PATCH`.
For upgrade instructions, please check the respective _Breaking Changes_ sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.16.4...HEAD)

### Breaking Changes
- Changed writing behavior. There is a new argument `allow_resize` for `MagView.write` and `View.write`, which defaults to `False`. If set to `True`, the bounding box of underlying `Layer` will be resized to fit the to-be-written data. That largely mirrors the previous behavior. However, it is not safe for concurrent operations, so it is disabled by default. It is recommended to set the `Layer.bounding_box` to the desired size before writing.
- Removed deprecated functions, properties and arguments:
  - Functions:
    - `open_annotation`, use `Annotation.load()` instead
    - `Dataset.get_color_layer`, use `Dataset.get_color_layers()` instead
    - `Dataset.get_segmentation_layer`, use `Dataset.get_segmentation_layers()` instead
    - `Dataset.create`, use `Dataset.__init__` instead
    - `Dataset.get_or_create`, use `Dataset.__init__` with `exist_ok=True` instead
    - `Layer.get_best_mag`, use `Layer.get_finest_mag` instead
    - `View.read_bbox`, use `read` with `relative_bounding_box` or `absolute_bounding_box` instead
    - `View.__enter__` and `View.__exit__`, context managers are not needed anymore
    - `open_nml`, use `Skeleton.load()` instead
    - `Group.add_graph`, use `Group.add_tree` instead
    - `Group.get_max_graph_id`, use `Group.get_max_tree_id` instead
    - `Group.flattened_graphs`, use `Group.flattened_trees` instead
    - `Group.get_graph_by_id`, use `Group.get_tree_by_id` instead
    - `Skeleton.from_path`, use `Skeleton.load()` instead
    - `Skeleton.write`, use `Skeleton.save()` instead
  - Properties:
    - `Annotation.username`, use `Annotation.owner_name` instead
    - `Annotation.scale`, use `Annotation.voxel_size` instead
    - `Annotation.user_id`, use `Annotation.owner_id` instead
    - `ArrayInfo.shard_size`, use `ArrayInfo.shard_shape` instead
    - `Dataset.scale`, use `Dataset.voxel_size` instead
    - `MagView.global_offset`, always `(0, 0, 0, ...)`
    - `MagView.size`, use `mag_view.bounding_box.in_mag(mag_view.mag).bottomright`
    - `MagViewProperties.resolution`, use `MagViewProperties.mag` instead
    - `LayerProperties.resolutions`, use `LayerProperties.mags` instead
    - `View.header`, use `View.info` instead
    - `View.global_offset`, use `view.bounding_box.in_mag(view.mag).topleft` instead
    - `View.size`, use `view.bounding_box.in_mag(view.mag).size` instead
    - `Group.graphs`, use `Group.trees`
    - `Skeleton.scale`, use `Skeleton.voxel_size` instead
  - Arguments:
    - `annotation_type` in `Annotation.download`, not needed anymore
    - `annotation_type` in `Annotation.open_as_remote_dataset`, not needed anymore
    - `size` in `BufferedSliceReader.__init__`, use `relative_bounding_box` or `absolute_bounding_box` instead
    - `offset` in `BufferedSliceReader.__init__`, use `relative_bounding_box` or `absolute_bounding_box` instead
    - `offset` in `BufferedSliceWriter.__init__`, use `relative_bounding_box` or `absolute_bounding_box` instead
    - `json_update_allowed` in `BufferedSliceWriter.__init__`, not supported anymore
    - `offset` in `BufferedSliceWriter.reset_offset`, use `relative_offset` or `absolute_offset` instead
    - `scale` in `Dataset.__init__`, use `voxel_size` or `voxel_size_with_unit` instead
    - `dtype` in `Dataset.add_layer`, use `dtype_per_channel` instead
    - `dtype` in `Dataset.get_or_add_layer`, use `dtype_per_channel` instead
    - `chunk_size` in `Dataset.add_layer_from_images`, use `chunk_shape` instead
    - `chunk_size` in `Dataset.copy_dataset`, use `chunk_shape` instead
    - `block_len` in `Dataset.copy_dataset`, use `chunk_shape` instead
    - `file_len` in `Dataset.copy_dataset`, use `chunks_per_shard` instead
    - `args` in `Dataset.copy_dataset`, use `executor` instead
    - `chunk_size` in `Layer.add_mag`, use `chunk_shape` instead
    - `block_len` in `Layer.add_mag`, use `chunk_shape` instead
    - `file_len` in `Layer.add_mag`, use `chunks_per_shard` instead
    - `chunk_size` in `Layer.get_or_add_mag`, use `chunk_shape` instead
    - `block_len` in `Layer.get_or_add_mag`, use `chunk_shape` instead
    - `file_len` in `Layer.get_or_add_mag`, use `chunks_per_shard` instead
    - `args` in `Layer.downsample`, use `executor` instead
    - `args` in `Layer.downsample_mag`, use `executor` instead
    - `args` in `Layer.redownsample`, use `executor` instead
    - `args` in `Layer.downsample_mag_list`, use `executor` instead
    - `args` in `Layer.downsample_mag_list`, use `executor` instead
    - `buffer_edge_len` in `Layer.upsample`, use `buffer_shape` instead
    - `args` in `Layer.upsample`, use `executor` instead
    - `min_mag` in `Layer.upsample`, use `finest_mag` instead
    - `offset` in `MagView.write`, use `relative_offset`, `absolute_offset`, `relative_bounding_box`, or `absolute_bounding_box` instead
    - `args` in `MagView.compress`, use `executor` instead
    - `offset` in `View.write`, use `relative_offset`, `absolute_offset`, `relative_bounding_box`, or `absolute_bounding_box` instead
    - `offset` in `View.read`, use `relative_offset`, `absolute_offset`, `relative_bounding_box`, or `absolute_bounding_box` instead
    - `offset` in `View.get_view`, use `relative_offset`, `absolute_offset`, `relative_bounding_box`, or `absolute_bounding_box` instead
    - `offset` in `View.get_buffered_slice_writer`, use `relative_offset`, `absolute_offset`, `relative_bounding_box`, or `absolute_bounding_box` instead
    - `offset` in `View.get_buffered_slice_reader`, use `relative_bounding_box`, or `absolute_bounding_box` instead
    - `size` in `View.get_buffered_slice_reader`, use `relative_bounding_box`, or `absolute_bounding_box` instead
    - `chunk_size` in `View.for_each_chunk`, use `chunk_shape` instead
    - `source_chunk_size` in `View.for_zipped_chunks`, use `source_chunk_shape` instead
    - `target_chunk_size` in `View.for_zipped_chunks`, use `target_chunk_shape` instead
    - `args` in `View.content_is_equal`, use `executor` instead
  - Classes:
    - `Graph`, use `Tree` instead
- Changed defaults:
  - `exist_ok` in `Dataset.__init__` is now `False`
  - `compress` in `Dataset.from_images` is now `True`
  - `compress` in `Dataset.add_layer_from_images` is now `True`
  - `DEFAULT_DATA_FORMAT` is now `Zarr3`
  - `compress` in `Layer.add_mag` is now `True`
  - `compress` in `Layer.upsample` is now `True`
  - `buffer_size` in `View.get_buffered_slice_reader` is now computed from the shard shape
  - `buffer_size` in `View.get_buffered_slice_writer` is now computed from the shard shape
- Moved from positional argument to keyword-only argument:
  - `json_update_allowed` in `MagView.write`
  - `json_update_allowed` in `View.write`
- Added arguments:
  - `allow_resize` in `MagView.write` with default `False`
  - `allow_resize` in `View.write` with default `False`


### Added
- Added support for python 3.13. [#1240](https://github.com/scalableminds/webknossos-libs/pull/1240)

### Changed

### Fixed


## [0.16.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.16.4) - 2025-01-23
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.16.3...v0.16.4)

### Added
- Added `list_bounding_boxes()` for Zarr-based datasets. [#1238](https://github.com/scalableminds/webknossos-libs/pull/1238)


## [0.16.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.16.3) - 2025-01-21
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.16.2...v0.16.3)

### Breaking Changes
- `RemoteDataset.display_name` is deprecated. To change the name of a dataset use the `name` property instead.
- `Dataset.get_remote_datasets()` returns a mapping. The keys of this mapping changed from datasets name to datasets id.
- `Task.create()` needs a `dataset_id` now instead of a `dataset_name`. Alternativly a `RemoteDataset` object can be used. The `dataset_name` is marked as deprecated. As `dataset_name` is an optional argument now its position has changed, this is important if `create()` is called only with positional arguments.
- When uploading an Annotation the organization_id is neccessary now. The organization_id might be stored in the Annotation object or it is inferred from the current webknossos_context. [#1155](https://github.com/scalableminds/webknossos-libs/pull/1155)

### Added
- `Dataset` method `get_remote_datasets()` accepts `name` and `folder_id` as arguments now to filter remote datasets.
- `RemoteDataset` got an additional property: `created`.
- `Annotation` got an additional property: `dataset_id`.
- `Dataset.trigger_dataset_import()` was added to refresh the datastore to register a newly added dataset. [#1236](https://github.com/scalableminds/webknossos-libs/pull/1236)

### Changed
- Updated to WEBKNOSSOS API version 9. This includes support for the new url structure for datasets and the usage of `dataset_id`. [#1231](https://github.com/scalableminds/webknossos-libs/pull/1231)

### Fixed
- Fixed Mag setup for non-public datasets. [#1222](https://github.com/scalableminds/webknossos-libs/pull/1222)
- Fixed an issue when shallow copying datasets with a remote mag. [#1224](https://github.com/scalableminds/webknossos-libs/pull/1224)



## [0.16.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.16.2) - 2024-12-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.16.1...v0.16.2)

### Breaking Changes
- `MagView.get_zarr_array` now returns a `tensorstore` array instead of a `zarr-python` array. [#1174](https://github.com/scalableminds/webknossos-libs/pull/1174)

### Changed
- Updated to WEBKNOSSOS API version 8. [#1185](https://github.com/scalableminds/webknossos-libs/pull/1185)
- Using tensorstore for reading and writing zarr 2 and 3 arrays. Removed `zarrita` and `zarr` dependency. [#1174](https://github.com/scalableminds/webknossos-libs/pull/1174)


## [0.16.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.16.1) - 2024-12-05
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.16.0...v0.16.1)

### Added
- Added .nrrd and .nhdr to supported suffixes. [#1228](https://github.com/scalableminds/webknossos-libs/pull/1228)
- Added more docstrings for many public classes and methods. [#1225](https://github.com/scalableminds/webknossos-libs/pull/1225)

### Changed
- Removes vcr-py from developer dependencies for testing and adds proxay for recording and replaying API requests. [#1198](https://github.com/scalableminds/webknossos-libs/pull/1198)
- Removed the CZI installation extra from `pip install webknossos[all]` by default. Users need to manually install it with `pip install --extra-index-url https://pypi.scm.io/simple/ webknossos[czi]`. [#1219](https://github.com/scalableminds/webknossos-libs/pull/1219)
- Refactored the PimsTiffReader to read the data directly from the tiff file without creating a memmap-able copy first. This greatly reduces the time and storage requirements for converting large tiff files. [#1212](https://github.com/scalableminds/webknossos-libs/pull/1212)

### Fixed
- Fixed an issue where adding existing trees to an annotation fails. [#1201](https://github.com/scalableminds/webknossos-libs/pull/1201)
- Fixed unpickling of the SSL_Context to allow for a second or third pickling. [#1223](https://github.com/scalableminds/webknossos-libs/pull/1223)
- Fixed offset error in upsample_cube job [#1209](https://github.com/scalableminds/webknossos-libs/pull/1209)


## [0.16.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.16.0) - 2024-11-27
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.11...v0.16.0)


## [0.15.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.11) - 2024-11-26
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.10...v0.15.11)

### Fixed
- Fixed pickling issue that has been introduced in 0.15.9. [#1218](https://github.com/scalableminds/webknossos-libs/pull/1218)


## [0.15.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.10) - 2024-11-25
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.9...v0.15.10)

### Fixed
- Fixed pickling issue that has been introduced in 0.15.9. [#1218](https://github.com/scalableminds/webknossos-libs/pull/1218)


## [0.15.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.9) - 2024-11-25
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.8...v0.15.9)

### Changed
Removed the CZI installation extra from `pip install webknossos[all]` by default. Users need to manually install it with `pip install --extra-index-url https://pypi.scm.io/simple/ webknossos[czi]`. [#1219](https://github.com/scalableminds/webknossos-libs/pull/1219)


## [0.15.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.8) - 2024-11-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.7...v0.15.8)

### Changed
- Fixed SSL certificate verification for remote datasets by adding CA certificates using `certifi`. [#1211](https://github.com/scalableminds/webknossos-libs/pull/1211)


## [0.15.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.7) - 2024-10-25
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.6...v0.15.7)

### Added
- Added `--coarsest-mag` argument to the `webknossos downsample` command. [#1203](https://github.com/scalableminds/webknossos-libs/pull/1203)

### Fixed
- Fixed an issue with merging annotations with compressed fallback layers.
- Fixed an issue where adding a Zarr array with other axes than `cxyz` leads to an error. [#1204](https://github.com/scalableminds/webknossos-libs/pull/1204)



## [0.15.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.6) - 2024-10-16
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.5...v0.15.6)

### Added
- Added `add_mag_from_zarrarray` to `Layer` class, to add existing Zarr arrays as a mag of a layer. [#1151](https://github.com/scalableminds/webknossos-libs/pull/1151)

### Changed
- Replaced the Python package manager `poetry` with `uv`. [#1199](https://github.com/scalableminds/webknossos-libs/pull/1199)


## [0.15.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.5) - 2024-09-26
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.4...v0.15.5)

### Added
- Webknossos API functions were added: `Team.get_list()`, `Team.add("new_name")`, `User.assign_team_roles("teamName", isTeamManager: True)` and `RemoteDataset.explore_and_add_remote()` are available now. [#1196](https://github.com/scalableminds/webknossos-libs/pull/1196)


## [0.15.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.4) - 2024-09-23
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.3...v0.15.4)

### Added
- Enable metadata access for remote datasets. [#1163](https://github.com/scalableminds/webknossos-libs/pull/1163)


## [0.15.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.3) - 2024-09-11
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.2...v0.15.3)

### Breaking Changes
- Conversion of images with 4 channels creates a dataset with four layers instead of a dataset with one RGB layer. [#1192](https://github.com/scalableminds/webknossos-libs/pull/1192)

### Changed
- Updated tifffile dependency to v2024.8.30. [#1190](https://github.com/scalableminds/webknossos-libs/pull/1190)


## [0.15.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.2) - 2024-09-05
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.1...v0.15.2)

### Changed
- Updated and clarified documentation for nd_bounding_box intersected_with method

### Fixed
- Fixed an issue with cattrs v24.1.0. [#1184](https://github.com/scalableminds/webknossos-libs/pull/1184)



## [0.15.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.1) - 2024-08-13
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.15.0...v0.15.1)

### Changed
- Updates zarrita to 0.2.7. [#1169](https://github.com/scalableminds/webknossos-libs/pull/1169)


## [0.15.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.15.0) - 2024-08-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.26...v0.15.0)

### Breaking Changes
- Python version 3.8 is no longer officially supported. [#1068](https://github.com/scalableminds/webknossos-libs/pull/1068)

### Added
- Added example for scaling a skeleton. [#1147](https://github.com/scalableminds/webknossos-libs/pull/1147)

### Changed
- Added options `--layer-name` and `--mag` for compress command of the CLI. [#1141](https://github.com/scalableminds/webknossos-libs/pull/1141)
- Added options `--chunk-shape` and `--chunks-per-shard` for convert command of the CLI. [#1150](https://github.com/scalableminds/webknossos-libs/pull/1150)
- The `from_images` method of the `Dataset` supports directories and single files as `input_path` now. [#1152](https://github.com/scalableminds/webknossos-libs/pull/1152)
- Added support for python version 3.12. [#1068](https://github.com/scalableminds/webknossos-libs/pull/1068)
- The number of pixel limit for JPG conversion is disabled now. [#1154](https://github.com/scalableminds/webknossos-libs/pull/1154)
- Added option `--batch-size` to the convert command of the CLI. [#1158](https://github.com/scalableminds/webknossos-libs/pull/1158)

### Fixed
- Fixed issue with webknossos URL and context URL being considered different when opening a remote dataset due to trailing slashes. [#1137](https://github.com/scalableminds/webknossos-libs/pull/1137)
- Fix an issue where the remote folder was not found when the folder path query includes a trailing slash. [#1164](https://github.com/scalableminds/webknossos-libs/pull/1164)



## [0.14.26](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.26) - 2024-07-22
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.25...v0.14.26)

### Fixed
- Add a converter to the VoxelSize field `factor`, to ensure it is a tuple.



## [0.14.25](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.25) - 2024-07-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.24...v0.14.25)

### Added
- Added support for new voxel size that stores unit and updated to WEBKNOSSOS API version 7. [#1136](https://github.com/scalableminds/webknossos-libs/pull/1136)


## [0.14.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.24) - 2024-07-09
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.23...v0.14.24)

### Added
- Added an implementation of padded_with_margins for NDBoundingBox class. [#1120](https://github.com/scalableminds/webknossos-libs/pull/1120)
- Added a new method add_nx_graphs to skeleton.py which supports to add nx.Graphs to the Skeleton object. [#1130](https://github.com/scalableminds/webknossos-libs/pull/1130)

### Changed
- Removed additional logging messages during image conversion. [#1124](https://github.com/scalableminds/webknossos-libs/pull/1124)

### Fixed
- Fixed an issue where cube jobs upsampling, downsampling and compress failed when performed on more than 3 dimensions. [#1095](https://github.com/scalableminds/webknossos-libs/pull/1095)
- Fixed an issue where webknossos libs crash when installed with minimal dependencies. [#1104](https://github.com/scalableminds/webknossos-libs/pull/1104)



## [0.14.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.23) - 2024-06-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.22...v0.14.23)

### Breaking Changes
- Preferring a custom tiff reader over the default PIMS reader to convert tiff files. This change enables the recognition of axis information and the support of tifffiles with more than 3 dimensions. However, it also leads to changed behavior when converting tiff files. Tiffs with axes other than c, x, y, and z, with a shape bigger than 1, are no longer supported for conversion to WKW. Please convert these files to Zarr or Zarr3 Datasets instead. [#1043](https://github.com/scalableminds/webknossos-libs/pull/1043)

### Added
- Added a pixel level heuristic for distinguishing color and segmentation layers when importing image data with the `from_images` or `add_layer_from_images` method. [#1007](https://github.com/scalableminds/webknossos-libs/pull/1007)
- Added .ims as supported suffix. [#1085](https://github.com/scalableminds/webknossos-libs/pull/1085)
- Added suffixes supported by bioformats for Zeiss CZI, Leica LOF, Zeiss LSM (laser scanning microscope), Zeiss LSM (Laser Scanning Microscope) 510/710, Leica XLEF and Zeiss AxioVision ZVI (Zeiss Vision Image). [#1086](https://github.com/scalableminds/webknossos-libs/pull/1086)
- Added suport for setting a default ID mapping for segmentation layers. [#1118](https://github.com/scalableminds/webknossos-libs/pull/1118)

### Changed
- Moved functional parts of merge volume annotation CLI to Dataset and Annotation classes. [#1055](https://github.com/scalableminds/webknossos-libs/pull/1055)
- Set a new max value for test_align_with_mag_against_numpy_implementation to avoid failures due to high numbers. [#1082](https://github.com/scalableminds/webknossos-libs/pull/1082)
- Updated dependabot.yml [#1087](https://github.com/scalableminds/webknossos-libs/pull/1087)
- Make lookup for supported suffixes case-insensitive. [#1100](https://github.com/scalableminds/webknossos-libs/pull/1100)

### Fixed
- Fixed an issue with downloading annotations through the Command Line Interface. [#1083](https://github.com/scalableminds/webknossos-libs/pull/1083)



## [0.14.22](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.22) - 2024-05-13
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.21...v0.14.22)

### Fixed
- Performing `webknossos upload` on a windows machine led to loss of directory structure due to backslashes in the relative paths. This was fixed by [#1067](https://github.com/scalableminds/webknossos-libs/pull/1067)



## [0.14.21](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.21) - 2024-05-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.20...v0.14.21)

### Changed
- Added `layer_name` as optional argument to `Dataset.from_images` method. If the created dataset contains only a single layer, `layer_name` is used, otherwise the given `layer_name` is a common prefix for all layers. [#1054](https://github.com/scalableminds/webknossos-libs/pull/1054)
- The context variable of View.get_buffered_slice_writer() is a BufferedSliceWriter now instead of a Generator. Interaction with the SliceWriter does not change, but updating the offset after first initialization is possible now. [#1052](https://github.com/scalableminds/webknossos-libs/pull/1052)


## [0.14.20](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.20) - 2024-04-23
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.19...v0.14.20)

### Changed
- Updated ruff to v0.4.0 [#1047](https://github.com/scalableminds/webknossos-libs/pull/1047)
- Added NIfTI suffix .nii to list of supported bioformats suffixes. [#1048](https://github.com/scalableminds/webknossos-libs/pull/1048)


## [0.14.19](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.19) - 2024-04-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.18...v0.14.19)

### Changed
- Removed special CLI command for cubing Nifti files. Use regular conversion command instead. Further, moved Python dependencies for examples and dev dependencies into optional groups which are not installed by default. Install with `poetry install --with dev --with examples`. [#1024](https://github.com/scalableminds/webknossos-libs/pull/1024)


## [0.14.18](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.18) - 2024-04-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.17...v0.14.18)

### Fixed
- Fixed a bug, where using an unaligned topleft value for `add_layer_from_images` leads to corrupted data. [#1036](https://github.com/scalableminds/webknossos-libs/pull/1036)



## [0.14.17](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.17) - 2024-04-10
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.16...v0.14.17)

### Changed
- The characters `@` and `$` are allowed within layer names now. [#1034](https://github.com/scalableminds/webknossos-libs/pull/1034)


## [0.14.16](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.16) - 2024-04-04
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.15...v0.14.16)

### Added
- Add CLI tool for offline merging of zip annotations with fallback datasets. [#996](https://github.com/scalableminds/webknossos-libs/pull/996)
- Added support for converting new file formats such as DICOM, using the bioformats reader. [#1014](https://github.com/scalableminds/webknossos-libs/pull/1014)

### Changed
- The rules for naming the layers have been tightened to match the allowed layer names on webknossos. [#1016](https://github.com/scalableminds/webknossos-libs/pull/1016)
- Replaced PyLint linter + black formatter with Ruff for development. [#1013](https://github.com/scalableminds/webknossos-libs/pull/1013)
- The remote operations now use the WEBKNOSSOS API version 6. [#1018](https://github.com/scalableminds/webknossos-libs/pull/1018)
- The conversion of 4D Tiff files to a Zarr3 Dataset is possible. NDBoundingBoxes and VecInt classes are introduced to support working with more than 3 dimensions. [#966](https://github.com/scalableminds/webknossos-libs/pull/966)


## [0.14.15](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.15) - 2024-02-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.14...v0.14.15)


## [0.14.14](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.14) - 2024-01-12
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.13...v0.14.14)

### Added
- Added a method to the Datasets class that calculates a dataset's bounding box covering all layers. [#975](https://github.com/scalableminds/webknossos-libs/pull/975)


## [0.14.13](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.13) - 2024-01-02
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.12...v0.14.13)

### Fixed

- Fixed a bug in reading project info from webknossos using the api client for non-admins. [#972](https://github.com/scalableminds/webknossos-libs/pull/972)



## [0.14.12](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.12) - 2023-12-19
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.11...v0.14.12)

### Fixed
- Fixes that the buffered slice writer could overwrite data when writing less slices than buffer_size at an offset that is not aligned. [#973](https://github.com/scalableminds/webknossos-libs/pull/973)


## [0.14.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.11) - 2023-12-06
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.10...v0.14.11)


## [0.14.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.10) - 2023-12-04
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.9...v0.14.10)

### Fixed

- Fixed a bug in reading project info from webknossos using the api client. [#970](https://github.com/scalableminds/webknossos-libs/pull/970)



## [0.14.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.9) - 2023-11-29
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.8...v0.14.9)


## [0.14.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.8) - 2023-11-28
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.7...v0.14.8)


## [0.14.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.7) - 2023-11-17
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.6...v0.14.7)


## [0.14.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.6) - 2023-11-17
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.5...v0.14.6)


## [0.14.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.5) - 2023-11-08
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.4...v0.14.5)

### Changed
- Performance improvements for reading from and writing to sharded zarr3 datasets, also speeding up the automated tests [#963](https://github.com/scalableminds/webknossos-libs/pull/963)


## [0.14.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.4) - 2023-11-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.3...v0.14.4)

### Added
- The `DEFAULT_DATA_FORMAT` and `DEFAULT_CHUNKS_PER_SHARD` can now be set with the env variables `WK_DEFAULT_DATA_FORMAT` and `WK_DEFAULT_CHUNKS_PER_SHARD`
- A `Vec3Int` can now be initialized with a string containing an int or a tuple.

### Changed
- Upgrades mypy to 1.6. [#956](https://github.com/scalableminds/webknossos-libs/pull/956)
- Refactored the WEBKNOSSOS API client to no longer use generated client code. [#948](https://github.com/scalableminds/webknossos-libs/pull/948)



## [0.14.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.3) - 2023-10-19
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.2...v0.14.3)

### Fixed
- Fixes bug in FSStore creation when using local paths for zarr data format. [#955](https://github.com/scalableminds/webknossos-libs/pull/955)



## [0.14.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.2) - 2023-10-18
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.1...v0.14.2)


## [0.14.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.1) - 2023-10-13
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.0...v0.14.1)
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.14.0...HEAD)

### Added

- Adds support for Zarr3-based volume annotations as introduced in [webknossos#7288](https://github.com/scalableminds/webknossos/pull/7288). [#952](https://github.com/scalableminds/webknossos-libs/pull/952)

### Changed

- The `WK_USE_ZARRITA` env variable is no longer required. [`zarrita`](https://github.com/scalableminds/zarrita) is always installed and now the default for Zarr and Zarr3 datasets. [#950](https://github.com/scalableminds/webknossos-libs/issues/950)
- Updates various dependencies. [#943](https://github.com/scalableminds/webknossos-libs/pull/943)


## [0.14.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.14.0) - 2023-10-11

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.7...v0.14.0)

### Breaking Changes

- `wait_and_ensure_success` from `webknossos.utils` now requires an `executor` argument. [#943](https://github.com/scalableminds/webknossos-libs/pull/943)

### Changed

- Updates various dependencies. [#943](https://github.com/scalableminds/webknossos-libs/pull/943)

## [0.13.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.7) - 2023-10-07

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.6...v0.13.7)

### Fixed

- Fixed a bug in writing compressed data. [#942](https://github.com/scalableminds/webknossos-libs/pull/942)

## [0.13.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.6) - 2023-08-17

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.5...v0.13.6)

## [0.13.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.5) - 2023-08-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.4...v0.13.5)

### Added

- Added `task_type` property to `Task` class. [#938](https://github.com/scalableminds/webknossos-libs/pull/938)

### Fixed

- Fixed a bug where parallel access to the properties json leads to an JsonDecodeError in the webknossos CLI [#919](https://github.com/scalableminds/webknossos-libs/issues/919)

## [0.13.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.4) - 2023-08-14

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.3...v0.13.4)

### Breaking Changes

- Task/Project management: `open` tasks have been renamed to `pending`. Use `Task.status.pending_instance_count` instead of `Task.status.open_instance_count`. [#930](https://github.com/scalableminds/webknossos-libs/pull/930)

### Fixed

- Fixed an infinite loop in the mag calculation during anisotropic downsampling in situations where the target mag cannot possibly be reached while adhering to the anisotropic downsampling scheme. [#934](https://github.com/scalableminds/webknossos-libs/pull/934)

## [0.13.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.3) - 2023-08-08

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.2...v0.13.3)

### Added

- `View` has a `map_chunk` method now to run a function on each chunk and collect the results in a list.

### Changed

- As WEBKNOSSOS does not require the largest segment id. It is also not mandatory in the WEBKNOSSOS libs anymore. [#917](https://github.com/scalableminds/webknossos-libs/issues/917) The method `SegmentationLayer.refresh_largest_segment_id` was added to lookup the highest value in segmentation data and set `largest_segment_id` accordingly.
- The `convert` command of the cli now has a `--category` flag, to select the LayerCategoryType.

## [0.13.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.2) - 2023-07-26

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.1...v0.13.2)

### Changed

- The `convert` command of the cli now has a `--category` flag, to select the LayerCategoryType.

## [0.13.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.1) - 2023-07-17

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.13.0...v0.13.1)

### Changed

- The conversion folder structures to layer names does not allow slashes in the layer name anymore. [#918](https://github.com/scalableminds/webknossos-libs/issues/918)

### Fixed

- Fixed a bug where compression in add_layer_from_images uses too much memory [#900](https://github.com/scalableminds/webknossos-libs/issues/900)

## [0.13.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.13.0) - 2023-06-21

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.6...v0.13.0)

### Added

- Added `duration_in_seconds` and `modified` to `AnnotationInfo`. [#914](https://github.com/scalableminds/webknossos-libs/pull/914)
- Added [`zarrita`](https://github.com/scalableminds/zarrita) storage backend for arrays. zarrita supports Zarr v2 and v3 including sharding. To activate zarrita, the environment variable `WK_USE_ZARRITA` must be set. [#912](https://github.com/scalableminds/webknossos-libs/pull/912)
- Added a `Zarr3` data format which supports sharding. [#912](https://github.com/scalableminds/webknossos-libs/pull/912)

### Changed

- Integrated the `wkcuber` CLI tool into `webknossos` package. [#903](https://github.com/scalableminds/webknossos-libs/pull/903)
  - To get an overview of all webknossos subcommands type `webknossos --help`. If the usage of a single subcommand is of interest type `webknossos <subcommand> --help`
  - These commands were changed:
    - `python -m wkcuber`, `python -m wkcuber.convert_image_stack_to_wkw` -> `webknossos convert`
    - `python -m wkcuber.export_wkw_as_tiff` -> `webknossos export-wkw-as-tiff`
    - `python -m wkcuber.convert_knossos` -> `webknossos convert-knossos`
    - `python -m wkcuber.convert_raw` -> `webknossos convert-raw`
    - `python -m wkcuber.downsampling` -> `webknossos downsample`
    - `python -m wkcuber.compress` -> `webknossos compress`
    - `python -m wkcuber.check_equality` -> `webknossos check-equality`
  - There is one new command:
    - `webknossos upload` to upload a dataset to a WEBKNOSSOS server
  - These commands have been removed:
    - `python -m wkcuber.cubing`
    - `python -m wkcuber.tile_cubing`
    - `python -m wkcuber.metadata`
    - `python -m wkcuber.recubing`

### Fixed

- Fixed a bug where upsampling of a layer would fail, if the layer had a bounding box that doesn't align with the from_mag mag. [#915](https://github.com/scalableminds/webknossos-libs/pull/915)

## [0.12.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.6) - 2023-06-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.5...v0.12.6)

### Changed

- Upgrades `wkw`. [#911](https://github.com/scalableminds/webknossos-libs/pull/911)

## [0.12.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.5) - 2023-06-01

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.4...v0.12.5)

### Added

- Added support for Python 3.11. [#843](https://github.com/scalableminds/webknossos-libs/pull/843)

## [0.12.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.4) - 2023-05-25

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.3...v0.12.4)

### Added

- `Group.add_tree` now also accepts a tree object as a first parameter (instead of only a string). [#891](https://github.com/scalableminds/webknossos-libs/pull/891)
- `Group.remove_tree_by_id` was added. [#891](https://github.com/scalableminds/webknossos-libs/pull/891)

### Changed

- Upgrades `black`, `mypy`, `pylint`, `pytest`. [#873](https://github.com/scalableminds/webknossos-libs/pull/873)

### Fixed

- Fixed poetry build backend for new versions of Poetry. [#899](https://github.com/scalableminds/webknossos-libs/pull/899)
- Added axis_order fields for Zarr data format. [#902](https://github.com/scalableminds/webknossos-libs/issues/902)

## [0.12.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.3) - 2023-02-22

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.2...v0.12.3)

### Added

- Added support to import ImageJ Hyperstack tiff files via `Dataset.from_images` and `dataset.add_layer_from_images`. [#877](https://github.com/scalableminds/webknossos-libs/pull/877)

### Changed

- `Dataset.from_images` and `dataset.add_layer_from_images` now automatically convert big endian dtypes to their little endian counterparts by default. [#877](https://github.com/scalableminds/webknossos-libs/pull/877)

### Fixed

- Fixed reading czi files with non-zero axis offsets. [#876](https://github.com/scalableminds/webknossos-libs/pull/876)

## [0.12.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.2) - 2023-02-20

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.1...v0.12.2)
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.1...HEAD)

### Added

- Added `RemoteFolder` for assigning remote datasets to a WEBKNOSSOS folder. [#868](https://github.com/scalableminds/webknossos-libs/pull/868)

## [0.12.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.1) - 2023-02-16

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.12.0...v0.12.1)

### Added

- Added `read_only` parameter for `annotation.temporary_volume_layer_copy`. [#866](https://github.com/scalableminds/webknossos-libs/pull/866)

### Changed

- in Dataset.layer_from_images, layer names generated from multiple input image channels no longer contain equal signs, yielding better url safety. [#867](https://github.com/scalableminds/webknossos-libs/pull/867)

### Fixed

- Fixed a bug where some czi, dm3, dm4 images could not be converted to wkw due to a too-strict check. [#865](https://github.com/scalableminds/webknossos-libs/pull/865)
- Enforce `read_only` property of datasets also for down- and upsampling. [#866](https://github.com/scalableminds/webknossos-libs/pull/866)

## [0.12.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.12.0) - 2023-02-10

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.11.4...v0.12.0)

### Breaking Changes

- Dropped support for Python 3.7. [#833](https://github.com/scalableminds/webknossos-libs/pull/833)

### Added

- Added support for Python 3.10. [#833](https://github.com/scalableminds/webknossos-libs/pull/833)

## [0.11.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.11.4) - 2023-02-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.11.3...v0.11.4)

## [0.11.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.11.3) - 2023-02-06

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.11.2...v0.11.3)

## [0.11.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.11.2) - 2023-01-18

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.11.1...v0.11.2)

### Fixed

- Fixed a bug, where the image order could be randomized when passing a directory path to `Dataset.add_layer_from_images`. [#854](https://github.com/scalableminds/webknossos-libs/pull/854)

## [0.11.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.11.1) - 2023-01-05

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.11.0...v0.11.1)

### Added

- `Dataset.from_images` and `dataset.add_layer_from_images` have new features: [#842](https://github.com/scalableminds/webknossos-libs/pull/842)
  - `dm3` and `dm4` datasets can be read without bioformats now.
  - It's possible to completely disable the bioformats adapter by setting `use_bioformats` to False.
  - Lists of images can now be handled with other readers, before only images supported by skimage worked in lists.

## [0.11.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.11.0) - 2022-12-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.27...v0.11.0)

### Breaking Changes

- Removed the `id` attribute of the `BoundingBox` class, also from the constructor. [#836](https://github.com/scalableminds/webknossos-libs/pull/836)

### Fixed

- Fixed bounding box serialization in NMLs, so that bounding boxes which are uploaded via annotations are now recognized properly by webKnossos. [#836](https://github.com/scalableminds/webknossos-libs/pull/836)
- Bounding boxes keep their name, color and visibility when transformed via methods, such as `bbox.padded_with_margins()`. [#836](https://github.com/scalableminds/webknossos-libs/pull/836)

## [0.10.27](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.27) - 2022-12-07

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.26...v0.10.27)

### Added

- Short links, such as `https://webknossos.org/links/93zLg9U9vJ3c_UWp`, are now supported for dataset and annotation urls in `download` and `open_remote` methods. [#837](https://github.com/scalableminds/webknossos-libs/pull/837)

## [0.10.26](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.26) - 2022-12-05

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.25...v0.10.26)

### Changed

- `Dataset.from_images` and `dataset.add_layer_from_images` now try to import the images via the [bioformats](https://www.openmicroscopy.org/bio-formats) after all other options as well. [#829](https://github.com/scalableminds/webknossos-libs/pull/829)

### Fixed

- `dataset.add_layer_from_images` can now handle paths to folders which only contain a single image. [#829](https://github.com/scalableminds/webknossos-libs/pull/829)

## [0.10.25](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.25) - 2022-11-29

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.24...v0.10.25)

### Breaking Changes

- `Dataset.from_images` now adds a layer per timepoint and per channel (if the data doesn't have 1 or 3 channels). [#822](https://github.com/scalableminds/webknossos-libs/pull/822)

### Added

- Added python-native CZI support for `Dataset.from_images` or `dataset.add_layer_from_images`, without using bioformats. [#822](https://github.com/scalableminds/webknossos-libs/pull/822)
- `dataset.add_layer_from_images` can add a layer per timepoint and per channel when passing `allow_multiple_layers=True`. [#822](https://github.com/scalableminds/webknossos-libs/pull/822)

## [0.10.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.24) - 2022-11-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.23...v0.10.24)

### Changed

- Updated cattrs dependency to 22.2.0. [#819](https://github.com/scalableminds/webknossos-libs/pull/819)

## [0.10.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.23) - 2022-11-01

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.22...v0.10.23)

### Breaking Changes

- `ds.add_layer_from_images`: Turned some arguments into keyword-only arguments, only affecting positional arguments after the first 8 arguments. [#818](https://github.com/scalableminds/webknossos-libs/pull/818)

### Added

- `ds.add_layer_from_images`: added topleft and dtype kw-only arguments. [#818](https://github.com/scalableminds/webknossos-libs/pull/818)

## [0.10.22](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.22) - 2022-10-27

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.21...v0.10.22)

### Fixed

- Fixed a bug where some image sequences could not be read in layer_from_images. [#817](https://github.com/scalableminds/webknossos-libs/pull/817)

## [0.10.21](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.21) - 2022-10-26

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.20...v0.10.21)

## [0.10.20](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.20) - 2022-10-20

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.19...v0.10.20)

### Changed

- Make volume locations optional, allowing to parse segment information in future NML-only annotations. [#814](https://github.com/scalableminds/webknossos-libs/pull/814)

### Fixed

- `annotation.temporary_volume_layer_copy()` works also with empty volume annotations. [#814](https://github.com/scalableminds/webknossos-libs/pull/814)

## [0.10.19](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.19) - 2022-10-18

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.18...v0.10.19)

### Breaking Changes

- The `executor` argument can now be passed to the following methods,
  `args` is deprecated now for those [#805](https://github.com/scalableminds/webknossos-libs/pull/805):
  - `dataset.copy_dataset(…)`
  - `layer.upsample(…)`
  - `layer.downsample(…)`
  - `layer.downsample_mag(…)`
  - `layer.downsample_mag_list(…)`
  - `layer.redownsample(…)`
  - `mag_view.compress(…)`
  - `view.content_is_equal(…)`

### Added

- Added `Dataset.from_images`, which converts images to a Dataset, possibly consisting of multiple layers. [#808](https://github.com/scalableminds/webknossos-libs/pull/808
- Added `Annotation.open_as_remote_dataset(…)`, which is a shorthand for `Annotation.download(...).get_remote_annotation_dataset()`.
  [#811](https://github.com/scalableminds/webknossos-libs/pull/811)
- `skeleton.save()` now also accepts paths with a `.zip` suffix. [#811](https://github.com/scalableminds/webknossos-libs/pull/811)
- Added `annotation.get_volume_layer_segments()` to interact with information from the `Segments` tab in annotations. This method returns a dict from segment ids to an object containing optional segment `name`, `color` and `anchor_position`. [#812](https://github.com/scalableminds/webknossos-libs/pull/812)
- Added convenience methods `Dataset.compress` and `Dataset.downsample` for compressing and downsampling all layers and mags in a dataset. [#813](https://github.com/scalableminds/webknossos-libs/pull/813)
- Added examples for downloading segment masks from webKnossos and cubing & uploading tiff stacks. [#813](https://github.com/scalableminds/webknossos-libs/pull/813)

## [0.10.18](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.18) - 2022-09-29

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.17...v0.10.18)

### Added

- `Annotation.download()` now accepts the keyword-only argument `skip_volume_data`, which can be set to `True` to omit downloading volume data. [#806](https://github.com/scalableminds/webknossos-libs/pull/806)

## [0.10.17](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.17) - 2022-09-26

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.16...v0.10.17)

### Added

- Added `annotation.get_remote_annotation_dataset()` to get a streamed annotation dataset, which also reflects fallback layers or applied mappings. [#794](https://github.com/scalableminds/webknossos-libs/pull/794)
- Added direct access to an underlying Zarr array with the `MagView.get_zarr_array()` method. [#792](https://github.com/scalableminds/webknossos-libs/pull/792)

## [0.10.16](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.16) - 2022-09-13

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.14...v0.10.16)

### Added

- Added direct access to an underlying Zarr array with the `MagView.get_zarr_array()` method. [#792](https://github.com/scalableminds/webknossos-libs/pull/792)

### Changed

- Upgraded `zarr` and `numcodecs`. [#798](https://github.com/scalableminds/webknossos-libs/pull/798)

## [0.10.14](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.14) - 2022-08-30

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.13...v0.10.14)

### Changed

- `dataset.add_copy_layer()` and `layer.add_copy_mag()` now read and write the image data, not copying files. This allows to stream data from remote datasets. To continue using the filesystem copy mechanism, please use `dataset.add_fs_copy_layer()` or `layer.add_fs_copy_mag()`. [#790](https://github.com/scalableminds/webknossos-libs/pull/790)

## [0.10.13](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.13) - 2022-08-22

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.12...v0.10.13)

### Breaking Changes

- Renamed `chunk_size` arguments to `chunk_shape`. `chunk_size` is still available as keyword-only argument, but deprecated. [#706](https://github.com/scalableminds/webknossos-libs/pull/706)

### Changed

- The largest_segment_id is optional now. [#786](https://github.com/scalableminds/webknossos-libs/pull/786)

## [0.10.12](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.12) - 2022-08-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.11...v0.10.12)

### Fixed

- Fixed `task.get_project()`. [#785](https://github.com/scalableminds/webknossos-libs/pull/785)

## [0.10.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.11) - 2022-08-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.10...v0.10.11)

## [0.10.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.10) - 2022-07-26

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.9...v0.10.10)

## [0.10.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.9) - 2022-07-22

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.8...v0.10.9)

## [0.10.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.8) - 2022-07-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.7...v0.10.8)

## [0.10.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.7) - 2022-07-14

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.6...v0.10.7)

### Breaking Changes

- The `Annotation` constructor takes the `owner_name` argument instead of `username`. This is only important when using keyword arguments. The `username` attributes are still available as a proxy for the `owner_name` attribute, but deprecated. [#760](https://github.com/scalableminds/webknossos-libs/pull/760)
- `user_id` on `AnnotationInfo` objects is deprecated, please use `owner_id` instead. [#760](https://github.com/scalableminds/webknossos-libs/pull/760)
- When self-hosting a webKnossos server, please note that a webknossos version >= `22.06.0` is required. [#760](https://github.com/scalableminds/webknossos-libs/pull/760) & [#764](https://github.com/scalableminds/webknossos-libs/pull/764)

### Added

- `Dataset.upload()` accepts `Layer` objects from a `RemoteDataset` in the `layers_to_link` argument list. Also, `LayerToLink` can consume those via `LayerToLink.from_remote_layer()`. [#761](https://github.com/scalableminds/webknossos-libs/pull/761)
- `Task.create()` accepts a `RemoteDataset` for the `dataset_name` argument. [#761](https://github.com/scalableminds/webknossos-libs/pull/761)
- Added `annotation.get_remote_base_dataset()` returning a `RemoteDataset`. [#761](https://github.com/scalableminds/webknossos-libs/pull/761)
- Added `Team.get_by_name()`. [#763](https://github.com/scalableminds/webknossos-libs/pull/763)
- Added `Dataset.get_remote_datasets()`. [#763](https://github.com/scalableminds/webknossos-libs/pull/763)

### Changed

- If a token is requested from the user on the commandline, it is now stored in the current context. Before, it was discarded. [#761](https://github.com/scalableminds/webknossos-libs/pull/761)
- `Annotation.download()` does not need the `annotation_type` anymore, and the type can also be omitted from passed URLs. [#764](https://github.com/scalableminds/webknossos-libs/pull/764)
- `Dataset.add_layer_from_images` allows smaller batch sizes for uncompressed writes. [#766](https://github.com/scalableminds/webknossos-libs/pull/766)
- `Dataset.add_layer_from_images` uses multiprocessing by default. [#766](https://github.com/scalableminds/webknossos-libs/pull/766)

### Fixed

- Fixed the bounding box inferral for volume annotation layers that were not saved in Mag(1). [#765](https://github.com/scalableminds/webknossos-libs/pull/765)

## [0.10.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.6) - 2022-06-27

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.5...v0.10.6)

## [0.10.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.5) - 2022-06-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.4...v0.10.5)

## [0.10.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.4) - 2022-06-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.3...v0.10.4)

### Breaking Changes

- Added an `align_with_other_layers` parameter to `Layer.downsample` & `layer.upsample`. When set to true (default), the magnifications of the existing dataset are used as guidance for downsampling/upsampling. Instead of passing a boolean, one can also pass another dataset that should be used as guidance. [#730](https://github.com/scalableminds/webknossos-libs/pull/730)
- Changed the name of `max_mag` in `Layer.downsample` to `coarsest_mag`. [#730](https://github.com/scalableminds/webknossos-libs/pull/730)

### Added

- Added `Dataset.add_layer_from_images()` to convert image stacks to wkw or zarr Dataset.
  This needs pims and possibly more packages, which can be installed using extras, e.g. "webknossos[all]".
  [#741](https://github.com/scalableminds/webknossos-libs/pull/741)

### Changed

- The `Layer.downsample` and `Layer.upsample` function now automatically downsample according to magnifications already existing in the dataset. This behaviour can be turned off by setting the new parameter `align_with_other_layers` to `False`. [#730](https://github.com/scalableminds/webknossos-libs/pull/730)
- `View.get_buffered_slice_reader()` and `View.get_buffered_slice_writer()` don't log anything by default now.
  To get the previous logging, please invoke them with `logging=True`.
  [#741](https://github.com/scalableminds/webknossos-libs/pull/741)

## [0.10.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.3) - 2022-06-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.2...v0.10.3)

### Added

- Added export of [OME-NGFF v0.4 metadata](https://ngff.openmicroscopy.org/0.4/) for all `Dataset`s that have a Zarr layer [#737](https://github.com/scalableminds/webknossos-libs/pull/737)

## [0.10.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.2) - 2022-05-20

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.1...v0.10.2)

### Changed

- Added Python 3.9 support to wk-libs [#716](https://github.com/scalableminds/webknossos-libs/pull/716)

### Fixed

- URLs for the webknossos-context (e.g. in the `WK_URL` env var or via `webknossos_context(url=…)`) may now contain `/` in the end and are sanitized. Before, requests would fail if the URL contained a final `/`. [#733](https://github.com/scalableminds/webknossos-libs/pull/733)

## [0.10.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.1) - 2022-05-10

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.10.0...v0.10.1)

## [0.10.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.10.0) - 2022-05-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.24...v0.10.0)

### Breaking Changes

- `Dataset.upload()` now returns a `RemoteDataset` instead instead of the URL string. You can get the URL via `remote_ds.url`. [#723](https://github.com/scalableminds/webknossos-libs/pull/723)
- `User.teams` now is a tuple instead of a list. [#723](https://github.com/scalableminds/webknossos-libs/pull/723)
- The deprecated `download_dataset` function now requires the `organization_id` argument. [#723](https://github.com/scalableminds/webknossos-libs/pull/723)

### Added

- Added `Dataset.open_remote()`, which returns an object of the new `RemoteDataset`. [#723](https://github.com/scalableminds/webknossos-libs/pull/723)
  This can
  - give the webknossos URL for the dataset as `remote_ds.url`,
  - read image data via the webknossos zarr interface, using the inherited `Dataset` methods, and
  - read and change the following dataset metadata as properties: `display_name`, `description`, `tags`, `is_public`, `sharing_token`, `allowed_teams`.
- `Team` instances also contain the `organization_id`. [#723](https://github.com/scalableminds/webknossos-libs/pull/723)

## [0.9.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.24) - 2022-05-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.23...v0.9.24)

### Fixed

- Fixed upsampling with constant z in certain anisotropic cases. [#720](https://github.com/scalableminds/webknossos-libs/pull/720)

## [0.9.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.23) - 2022-05-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.22...v0.9.23)

## [0.9.22](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.22) - 2022-05-02

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.21...v0.9.22)

### Breaking Changes

- Renamed some entities to comply with the [webKnossos terminology](https://docs.webknossos.org/webknossos/terminology.html). [#704](https://github.com/scalableminds/webknossos-libs/pull/704):
  - The `Graph` class changed to `Tree`, also related methods and attributes are renamed now, e.g. `add_graph` is now `add_tree`.
    All previous entities are still available, but deprecated.
  - `scale` has changed to `voxel_size` for datasets, skeletons and annotations.
    Changes in `Dataset` are backwards-compatible, but `scale` is deprecated.
    For `Annotation` and `Skeletons` the initializer only supports `voxel_size`, the `scale` attribute is deprecated
  - `get_best_mag` is deprecated, please use `get_finest_mag` instead
  - In `layer.upscale`, `min_mag` is deprecated in favor of `finest_mag`

### Fixed

- Correctly maintain default_view_configuration property when downloading a dataset. [#677](https://github.com/scalableminds/webknossos-libs/pull/677)

## [0.9.21](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.21) - 2022-04-20

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.20...v0.9.21)

## [0.9.20](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.20) - 2022-04-20

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.19...v0.9.20)

### Fixed

- Fixed a bug where the server’s error message during dataset upload was not displayed to the user. [#702](https://github.com/scalableminds/webknossos-libs/pull/702)

## [0.9.19](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.19) - 2022-04-11

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.18...v0.9.19)

## [0.9.18](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.18) - 2022-04-06

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.17...v0.9.18)

## [0.9.17](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.17) - 2022-04-05

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.16...v0.9.17)

## [0.9.16](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.16) - 2022-03-31

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.15...v0.9.16)

## [0.9.15](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.15) - 2022-03-31

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.14...v0.9.15)

### Added

- Added cloud storage support for the `Dataset` classes, by using `UPath` from [universal_pathlib](https://github.com/fsspec/universal_pathlib) and [fsspec](https://github.com/fsspec/fsspec). Create remote datasets like this `Dataset(UPath("s3://bucket/path/to/dataset", key="...", secret="..."), scale=(11, 11, 24))`. Datasets on cloud storage only work with [Zarr](https://zarr.dev/)-based layers. [#649](https://github.com/scalableminds/webknossos-libs/pull/649)

## [0.9.14](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.14) - 2022-03-24

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.13...v0.9.14)

### Breaking Changes

- `Dataset.download`: The argument `dataset_name` was renamed to `dataset_name_or_url`. [#660](https://github.com/scalableminds/webknossos-libs/pull/660)

### Added

- `Dataset.download` now also accepts a URL, as well as a `sharing_token`, which can also be part of the URL. [#660](https://github.com/scalableminds/webknossos-libs/pull/660)

## [0.9.13](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.13) - 2022-03-24

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.12...v0.9.13)

### Breaking Changes

- Consistently rename `organization_name` parameters to `organization_id` (except in deprecated `webknossos.client.download_dataset`). [#639](https://github.com/scalableminds/webknossos-libs/pull/639)

### Changed

- `MagView.compress` now skips in-place compression of already compressed mags. [#667](https://github.com/scalableminds/webknossos-libs/pull/667)
- Replaced uses of `pathlib.Path` with `UPath` from [universal_pathlib](https://github.com/fsspec/universal_pathlib). Since `UPath` is compatible with `pathlib.Path`, changes in user code are not necessary. [#649](https://github.com/scalableminds/webknossos-libs/pull/649)

### Fixed

- Fixed compression of downsampled mags for layers with arbitrary and potentially mag-unaligned bounding boxes. [#667](https://github.com/scalableminds/webknossos-libs/pull/667)

## [0.9.12](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.12) - 2022-03-18

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.11...v0.9.12)

### Breaking Changes

- The annotation class now exposes `BoundingBox` objects instead of tuples. [#646](https://github.com/scalableminds/webknossos-libs/pull/646)

### Added

- Added `groups` and `graphs` property to skeleton.Group to access immediate child groups/graphs. [#645](https://github.com/scalableminds/webknossos-libs/pull/645)
- The `BoundingBox` class now supports the following additional properties: `id`, `name`, `is_visible` and `color`. [#646](https://github.com/scalableminds/webknossos-libs/pull/646)
- Added support for [Zarr](https://zarr.dev/) arrays in the `Dataset` classes. Users can set the `data_format` of layers to `zarr` to use Zarr for storing data. [#627](https://github.com/scalableminds/webknossos-libs/pull/627)
  The current implementation has some limitations, e.g.:
  - Only one type of compression (Blosc+Zstd) is implemented.
  - Sharding is not available in Zarr, yet. Please use `chunks_per_shard = (1, 1, 1)`.
  - Only local filesystem-based arrays are supported.
    There are changes to the `datasource-properties.json` for Zarr layers compared to WKW layers:
  - `dataFormat` needs to be changed to `zarr`.
  - The list of mags is called `mags`, instead of `wkwResolutions`.
  - Each mag is represented by an object with a single attribute `mag`, e.g. `{ "mag": [1, 1, 1] }`.

### Changed

- Dataset: `block_len` and `file_len` attributes are now deprecated, but still available for backwards compatibility. Use `chunk_size` and `chunks_per_shard` instead. These new attributes are `Vec3Int`, so they can be set non-uniformly. However, WKW-backed layers still require uniform `chunk_size` and `chunks_per_shard`. [#627](https://github.com/scalableminds/webknossos-libs/pull/627)

### Fixed

- Fixed crash during downsampling and compression of segmentation layers. [#657](https://github.com/scalableminds/webknossos-libs/pull/657)

## [0.9.11](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.11) - 2022-03-16

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.10...v0.9.11)

## [0.9.10](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.10) - 2022-03-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.9...v0.9.10)

### Changed

- Annotation: `Annotation.temporary_volume_layer_copy` now uses the NML-provided `scale`. [#644](https://github.com/scalableminds/webknossos-libs/pull/644)
- Dataset: Moved the deprecation warning from `get_color_layers()` to the actually deprecated method `get_color_layer()`.
  [#635](https://github.com/scalableminds/webknossos-libs/pull/635)
- Inconsistent writes to datasets properties (e.g., caused due to multiprocessing) are detected automatically. The warning can be escalated to an exception with `warnings.filterwarnings("error", module="webknossos", message=r"\[WARNING\]")`. [#633](https://github.com/scalableminds/webknossos-libs/pull/633)
- Changed the `position` of a `skeleton.Node` to use `Vec3Int` instead of `(Float, Float, Float)`, because webKnossos stores node positions as integers. [#645](https://github.com/scalableminds/webknossos-libs/pull/645)

### Fixed

- Tests: The `./test.sh` script works on macOS again and doesn't throw Network Errors anymore. However the introduced fix could lead to slightly different behaviour on macOS tests vs CI tests, when UNIX socket communication is involved. [#618](https://github.com/scalableminds/webknossos-libs/pull/618)

## [0.9.9](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.9) - 2022-03-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.8...v0.9.9)

### Breaking Changes

- Changed the interface and behavior of `Annotation`s and `Skeleton`s, as well as nml-parsing related code.
  [#602](https://github.com/scalableminds/webknossos-libs/pull/602)
  - `webknossos.skeleton.nml` is not exposed anymore. The previous functionality may be found in
    `webknossos._nml` and `webknossos/annotation/_nml_conversion.py ` if needed, but please not that
    this is not part of the public API and may change at any time. Please use the respective logic on the
    `Annotation` class instead.
  - The `name` attribute on the `Skeleton` class changed to `dataset_name`.
  - The deprecated `Skeleton.offset` attribute is removed.
  - The following attributes are removed from the `Skeleton` class, instead they are part of the
    `Annotation` class now: `time`, `edit_position`, `edit_rotation`, `zoom_level`, `task_bounding_box`,
    `user_bounding_boxes`.
  - The following `Annotation` methods were renamed and their arguments changed slightly:
    - `save_volume_annotation` ➜ `export_volume_layer_to_dataset`
    - `temporary_volume_annotation_layer_copy` ➜ `temporary_volume_layer_copy`

### Added

- Added new features to the `Annotation` and `Skeleton` classes. [#602](https://github.com/scalableminds/webknossos-libs/pull/602)
  - The `Skeleton` class has new attributes `description` and `organization_id`.
  - The `Annotation` class has new attributes `username` and `annotation_id`, as well as the following
    attributes that are proxies for the skeleton attributes: `dataset_name`, `scale`, `organization_id`, `description`
  - `Annotation`s can now be initialized from their attributes and an optional skeleton.
  - New methods on `Annotation`: `upload`, `add_volume_layer`, `delete_volume_layer`
  - `Annotation.load()` and `annotation.save()` also works with `.nml` files.
- Added `MagView.get_views_on_disk()` as a shortcut to turning `get_bounding_boxes_on_disk` into views.
  [#621](https://github.com/scalableminds/webknossos-libs/pull/621)

### Fixed

- Fixed the download of skeleton-only annotations. [#602](https://github.com/scalableminds/webknossos-libs/pull/602)

## [0.9.8](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.8) - 2022-02-28

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.7...v0.9.8)

### Added

- Added `allow_overwrite` parameter to `Layer.downsample()`. [#614](https://github.com/scalableminds/webknossos-libs/pull/614)

## [0.9.7](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.7) - 2022-02-28

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.6...v0.9.7)

### Added

- Added `only_setup_mag` parameter to downsample-related methods in `Layer`. This parameter allows creating output magnifications before parallelizing downsampling invocations to avoid outdated writes to datasource-properties.json. [#610](https://github.com/scalableminds/webknossos-libs/pull/610)
- Added `Task.create()` method to create tasks by providing a dataset name, location, and rotation. [#605](https://github.com/scalableminds/webknossos-libs/pull/605)

## [0.9.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.6) - 2022-02-15

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.5...v0.9.6)

### Added

- Added `apply_merger_mode.py` example. [#592](https://github.com/scalableminds/webknossos-libs/pull/592)
- Added support for reading from multiple volume layers in annotations. If an annotation contains multiple volume layers, the layer name has to be provided when reading from a volume layer in an annotation (in `Annotation.save_volume_annotation()` and `Annotation.temporary_volume_annotation_layer_copy()`). Also, added the method `Annotation.get_volume_layer_names()` to see available volume layers. [#588](https://github.com/scalableminds/webknossos-libs/pull/588)

### Changed

- Dataset writes in higher mags do not increase the bounding box if the written data fits in the rounded up box. [#595](https://github.com/scalableminds/webknossos-libs/pull/595)

### Fixed

- Dataset down- & upload: [#595](https://github.com/scalableminds/webknossos-libs/pull/595)
  - Fixed download of higher mags.
  - Bounding box after download is set correctly (was inflated before when downloading higher mags).
  - The returned URL for uploads is corrected, this did not respect `new_dataset_name` before.

## [0.9.5](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.5) - 2022-02-10

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.4...v0.9.5)

### Fixed

- Skeleton: Fixed a bug when comparing `Graph` instances, this fixes failing loads which had the error message `Can only compare wk.Graph to another wk.Graph.` before. [#593](https://github.com/scalableminds/webknossos-libs/pull/593)

## [0.9.4](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.4) - 2022-02-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.3...v0.9.4)

### Added

- Added AnnotationInfo, Project and Task classes for handling annotation information and annotation project administration. [#574](https://github.com/scalableminds/webknossos-libs/pull/574)

### Changed

- Lifted the restriction that `BoundingBox` cannot have a negative topleft (introduced in v0.9.0). Also, negative size dimensions are flipped, so that the topleft <= bottomright,
  e.g. `BoundingBox((10, 10, 10), (-5, 5, 5))` -> `BoundingBox((5, 10, 10), (5, 5, 5))`. [#589](https://github.com/scalableminds/webknossos-libs/pull/589)

## [0.9.3](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.3) - 2022-02-07

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.2...v0.9.3)

### Fixed

- `dataset.upload(layers_to_link=…)`: Fixed a bug where the upload did not complete if layers_to_link contained layers present in uploading dataset. [#584](https://github.com/scalableminds/webknossos-libs/pull/584)

## [0.9.2](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.2) - 2022-02-03

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.1...v0.9.2)

### Added

- A custom network request timeout can be set using `webknossos_context(…, timeout=300)` or `export WK_TIMEOUT="300"`. [#577](https://github.com/scalableminds/webknossos-libs/pull/577)

### Changed

- The default network request timeout changed from ½min to 30 min. [#577](https://github.com/scalableminds/webknossos-libs/pull/577)

## [0.9.1](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.1) - 2022-01-31

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.0...v0.9.1)

### Changed

- The signatures of `(Mag)View.for_each_chunk` and `(Mag)View.for_zipped_chunks` changed:
  - The previous argument `work_on_chunk` is now called `func_per_chunk`.
  - The various `chunk_size` arguments now have to be given in Mag(1). They now have default values.
- Deprecations in `(Mag)View.get_buffered_slice_reader/_writer` [#564](https://github.com/scalableminds/webknossos-libs/pull/564):
  - `(Mag)View.get_buffered_slice_reader`: using the parameters `offset` and `size` is deprecated.
    Please use the parameter relative_bounding_box or absolute_bounding_box (both in Mag(1)) instead.
    The old offset behavior was absolute for `MagView`s and relative for `View`s.
  - `(Mag)View.get_buffered_slice_writer`: using the parameter `offset` is deprecated.
    Please use the parameter relative_offset or absolute_offset (both in Mag(1)) instead.
    The old offset behavior was absolute for `MagView`s and relative for `View`s.

## [0.9.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.0) - 2022-01-19

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.31...v0.9.0)

### Breaking Changes

- Various changes in View & MagView signatures [#553](https://github.com/scalableminds/webknossos-libs/pull/553):
  - **Breaking Changes**:
    - `MagView.read`: if nothing is supplied and the layer does not start at (0, 0, 0),
      the default behaviour changes from starting at absolute (0, 0, 0) to the layer's bounding box
    - `MagView.write`: if no offset is supplied and the layer does not start at (0, 0, 0),
      the default behaviour changes from starting at absolute (0, 0, 0) to the layer's bounding box
    - `(Mag)View.get_view`: read_only is a keyword-only argument now
    - `MagView.get_bounding_boxes_on_disk()` now returns an iterator yielding bounding boxes in Mag(1)
    - `BoundingBox` cannot have negative topleft or size entries anymore (lifted in v0.9.4).
  - **Deprecations**
    The following usages are marked as deprecated with warnings and will be removed in future releases:
    - Using the `offset` parameter for `read`/`write`/`get_view` in MagView and View is deprecated.
      There are new counterparts `absolute_offset` and `relative_offset` which have to be specified in Mag(1),
      whereas `offset` previously was specified in the Mag of the respective View.
      Also, for `read`/`get_view` only using `size` is deprecated, since it used to refer to the size in the View's Mag.
      Instead, `size` should always be used together with `absolute_offset` or `relative_offset`. Then it is interpreted in Mag(1).
    - The (Mag)View attributes `view.global_offset` and `view.size` are deprecated now, which were in the Mag of the respective View.
      Please use `view.bounding_box` instead, which is in Mag(1).
    - `read_bbox` on the (Mag)View is deprecated as well, please use `read` with the `absolute_bounding_box`or `relative_bounding_box` parameter instead. You'll have to pass the bounding box in Mag(1) then.

### Added

- Added a check for dataset name availability before attempting to upload. [#555](https://github.com/scalableminds/webknossos-libs/pull/555)

### Fixed

- Fixed the dataset download of private datasets which need a token. [#562](https://github.com/scalableminds/webknossos-libs/pull/562)

## [0.8.31](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.31) - 2022-01-07

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.30...v0.8.31)

### Added

- Added `Annotation.save("file_name")` to save an annotation to a file and `Annotation.temporary_volume_annotation_layer_copy()` to read from the volume layer of an annotation as a WK dataset. [#528](https://github.com/scalableminds/webknossos-libs/pull/528)
- Added `layers_to_link` parameter to `Dataset.upload()` so that layers don't need to be uploaded again if they already exist in another dataset on webKnossos. [#544](https://github.com/scalableminds/webknossos-libs/pull/544)

## [0.8.30](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.30) - 2021-12-27

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.29...v0.8.30)

### Breaking Changes

- The BoundingBoxNamedTuple was removed. Use BoundingBox instead. [#526](https://github.com/scalableminds/webknossos-libs/pull/526)
- Some methods of creating, opening and saving have changed. The old methods are still available but deprecated. [The documentation gives a good overview](https://docs.webknossos.org/api/webknossos.html). Specifically, the changes are :
  - `Dataset.create()` → `Dataset()`
  - `Dataset.get_or_create()` → `Dataset(…, exist_ok=True)`
  - `Dataset()` → `Dataset.open()`
  - `download_dataset()` → `Dataset.download()`
  - `open_annotation()` → `Annotation.load()` for local files, `Annotation.download()` to download from webKnossos
  - `open_nml()` → `Skeleton.load()`
  - `Skeleton.from_path()` → `Skeleton.load()`
  - `Skeleton.write()` → `Skeleton.save()`
    The deprecated methods will be removed in future releases.
    [#520](https://github.com/scalableminds/webknossos-libs/pull/520)

### Changed

- The detailed output of e.g. downsampling was replaced with a progress bar. [#527](https://github.com/scalableminds/webknossos-libs/pull/527)
- Always use the sampling mode `CONSTANT_Z` when downsampling 2D data. [#516](https://github.com/scalableminds/webknossos-libs/pull/516)
- Make computation of `largestSegmentId` more efficient for volume annotations. [#531](https://github.com/scalableminds/webknossos-libs/pull/531)
- Consistently use resolved instead of absolute path if make_relative is False. [#536](https://github.com/scalableminds/webknossos-libs/pull/536)

## [0.8.29](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.29) - 2021-12-14

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.28...v0.8.29)

### Breaking Changes

- To download datasets, a recent webknossos server version is necessary (>= 21.12.0). webknossos.org is unaffected. [#510](https://github.com/scalableminds/webknossos-libs/pull/510)

## [0.8.28](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.28) - 2021-12-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.27...v0.8.28)

## [0.8.27](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.27) - 2021-12-09

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.25...v0.8.27)

## [v0.8.25](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.25) - 2021-12-07

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.24...v0.8.25)

### Added

- Added support to download datasets from external datastores, which is the case for webknossos.org. [#497](https://github.com/scalableminds/webknossos-libs/pull/497)

### Changed

- Adapt the dataset upload to new webKnossos api. [#484](https://github.com/scalableminds/webknossos-libs/pull/484)
- `get_segmentation_layer()` and `get_color_layer()` were deprecated and should not be used, anymore, as they will fail if no or more than one layer exists for each category. Instead, `get_segmentation_layers()` and `get_color_layers()` should be used (if desired in combination with `[0]` to get the old, error-prone behavior).
- Renamed the folder webknossos/script-collection to webknossos/script_collection to enable module imports. [#505](https://github.com/scalableminds/webknossos-libs/pull/505)

## [v0.8.24](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.24) - 2021-11-30

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.23...v0.8.24)

## [v0.8.23](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.23) - 2021-11-29

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.22...v0.8.23)

### Breaking Changes

- `wk.Graph` now inherits from `networkx.Graph` directly. Therefore, the `nx_graph` attribute is removed. [#481](https://github.com/scalableminds/webknossos-libs/pull/481)
- The class `LayerCategories` was removed. `COLOR_TYPE` and `SEGMENTATION_TYPE` were renamed to `COLOR_CATEGORY` and `SEGMENTATION_CATEGORY` and can now be imported directly. The type of many parameters were changed from `str` to the literal `LayerCategoryType`. [#454](https://github.com/scalableminds/webknossos-libs/pull/454)

### Added

- Added `redownsample()` method to `Layer` to recompute existing downsampled magnifications. [#461](https://github.com/scalableminds/webknossos-libs/pull/461)
- Added `globalize_floodfill.py` script to globalize partially computed flood fill operations. [#461](https://github.com/scalableminds/webknossos-libs/pull/461)

### Changed

- Improved performance for calculations with `Vec3Int` and `BoundingBox`. [#461](https://github.com/scalableminds/webknossos-libs/pull/461)

### Fixed

- Resolve path when symlinking layer and make_relative is False (instead of only making it absolute). [#492](https://github.com/scalableminds/webknossos-libs/pull/492)

## [0.8.22](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.22) - 2021-11-01

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.21...v0.8.22)

### Breaking Changes

- Removed the `organization` parameter from the `webknossos_context` function. The organization will automatically be fetched using the token of the user. [#470](https://github.com/scalableminds/webknossos-libs/pull/470)

### Fixed

- Make Views picklable. We now ignore the file handle when we pickle Views. [#469](https://github.com/scalableminds/webknossos-libs/pull/469)

## [v0.8.19](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.19) - 2021-10-21

[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.18...v.8.19)

### Added

- Added a `User` class to the client that can be used to get meta-information of users or their logged time. The currently logged in user can be accessed, as well as all managed users. [#470](https://github.com/scalableminds/webknossos-libs/pull/470)

## [0.8.21](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.21) - 2021-10-28

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.20...v0.8.21)

### Changed

- Downgraded typing-extensions for better dependency compatibility [#472](https://github.com/scalableminds/webknossos-libs/pull/472)

## [0.8.20](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.20) - 2021-10-28

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.19...v0.8.20)

### Breaking Changes

- `BoundingBox.chunk()`'s 2nd parameter `chunk_border_alignments` now does not accept a list with a single `int` anymore. [#452](https://github.com/scalableminds/webknossos-libs/pull/452)

### Fixed

- Make Views picklable. We now ignore the file handle when we pickle Views. [#469](https://github.com/scalableminds/webknossos-libs/pull/469)

## [0.8.19](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.19) - 2021-10-21

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.18...v0.8.19)

### Breaking Changes

- `View`s now always open the `wkw.Dataset` lazily. All explicit calls to `View.open()` and `View.close()` must be removed. [#448](https://github.com/scalableminds/webknossos-libs/pull/448)
-

### Added

- Added a new Annotation class which includes skeletons as well as volume-annotations. [#452](https://github.com/scalableminds/webknossos-libs/pull/452)
- Added dataset down- and upload as well as annotation download, see the examples `learned_segmenter.py` and `upload_image_data.py`. [#452](https://github.com/scalableminds/webknossos-libs/pull/452)

## [0.8.18](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.18) - 2021-10-18

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.16...v0.8.18)

### Added

- The Dataset class now has a new method: add_shallow_copy. [#437](https://github.com/scalableminds/webknossos-libs/pull/437)

### Changed

- The `Vec3Int` constructor now asserts that its components are whole numbers also in numpy case. [#434](https://github.com/scalableminds/webknossos-libs/pull/434)
- Updated scikit-image dependency to 0.18.3. [#435](https://github.com/scalableminds/webknossos-libs/pull/435)
- `BoundingBox.contains` now also takes float points in numpy arrays. [#450](https://github.com/scalableminds/webknossos-libs/pull/450)

### Fixed

## [0.8.16](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.16) - 2021-09-22

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.15...v0.8.16)

### Breaking Changes

- Refactored the BufferedSliceWriter and added a BufferedSliceReader. [#425](https://github.com/scalableminds/webknossos-libs/pull/425)
  - BufferedSliceWriter
    - The data no longer gets transposed: previously the format of the slices was [y,x]; now it is [x,y]
    - The interface of the constructor was changed:
      - A `View` (or `MagView`) is now required as datasource
      - The parameter `dimension` can be used to specify the axis along the data is sliced
      - The offset is expected to be in the magnification of the view
    - This class is now supposed to be used within a context manager and the slices are written by sending them to the generator (see documentation of the class).
  - BufferedSliceReader
    - This class was added complementary to the BufferedSliceWriter
  - Added methods to get a BufferedSliceReader/BufferedSliceWriter from a View directly

### Added

### Changed

### Fixed

## [0.8.15](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.15) - 2021-09-22

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.13...v0.8.15)

### Breaking Changes

- Breaking changes were introduced for geometry classes in [#421](https://github.com/scalableminds/webknossos-libs/pull/421):

  - `BoundingBox`

    - is now immutable, use convenience methods, e.g. `bb.with_topleft((0,0,0))`
    - properties topleft and size are now Vec3Int instead of np.array, they are each immutable as well
    - all `to_`-conversions return a copy, some were renamed:
    - `to_array` → `to_list`
    - `as_np` → `to_np`
    - `as_wkw` → `to_wkw_dict`
    - `from_wkw` → `from_wkw_dict`
    - `as_config` → `to_config_dict`
    - `as_checkpoint_name` → `to_checkpoint_name`
    - `as_tuple6` → `to_tuple6`
    - `as_csv` → `to_csv`
    - `as_named_tuple` → `to_named_tuple`
    - `as_slices` → `to_slices`
    - `copy` → (gone, immutable)

  - `Mag`
    - is now immutable
    - `mag.mag` is now `mag._mag` (considered private, use to_list instead if you really need it as list)
    - all `to_`-conversions return a copy, some were renamed:
    - `to_array` → `to_list`
    - `scale_by` → (gone, immutable)
    - `divide_by` → (gone, immutable)
    - `as_np` → `to_np`

### Added

- An immutable Vec3Int class was introduced that holds three integers and provides a number of convenience methods and accessors. [#421](https://github.com/scalableminds/webknossos-libs/pull/421)

### Changed

- `BoundingBox` and `Mag` are now immutable attr classes containing `Vec3Int` values. See breaking changes above.

### Fixed

-

## [0.8.13](https://github.com/scalableminds/webknossos-cuber/releases/tag/v0.8.13) - 2021-09-22

[Commits](https://github.com/scalableminds/webknossos-cuber/compare/v0.8.12...v0.8.13)

This is the latest release at the time of creating this changelog.
