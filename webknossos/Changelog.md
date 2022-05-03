# Changelog

All notable changes to the webknossos python library are documented in this file.

Please see the [Stability Policy](./stability_policy.md) for details about the version schema
and compatibility implications.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/) `MAJOR.MINOR.PATCH`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.24...HEAD)

### Breaking Changes

### Added

### Changed

### Fixed


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
  * The `Graph` class changed to `Tree`, also related methods and attributes are renamed now, e.g. `add_graph` is now `add_tree`.
    All previous entities are still available, but deprecated.
  * `scale` has changed to `voxel_size` for datasets, skeletons and annotations.
    Changes in `Dataset` are backwards-compatible, but `scale` is deprecated.
    For `Annotation` and `Skeletons` the initializer only supports `voxel_size`, the `scale` attribute is deprecated
  * `get_best_mag` is deprecated, please use `get_finest_mag` instead
  * In `layer.upscale`, `min_mag` is deprecated in favor of `finest_mag`

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
  * Only one type of compression (Blosc+Zstd) is implemented.
  * Sharding is not available in Zarr, yet. Please use `chunks_per_shard = (1, 1, 1)`.
  * Only local filesystem-based arrays are supported.
  There are changes to the `datasource-properties.json` for Zarr layers compared to WKW layers:
  * `dataFormat` needs to be changed to `zarr`.
  * The list of mags is called `mags`, instead of `wkwResolutions`.
  * Each mag is represented by an object with a single attribute `mag`, e.g. `{ "mag": [1, 1, 1] }`.

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
  * `webknossos.skeleton.nml` is not exposed anymore. The previous functionality may be found in
    `webknossos._nml` and `webknossos/annotation/_nml_conversion.py ` if needed, but please not that
    this is not part of the public API and may change at any time. Please use the respective logic on the
    `Annotation` class instead.
  * The `name` attribute on the `Skeleton` class changed to `dataset_name`.
  * The deprecated `Skeleton.offset` attribute is removed.
  * The following attributes are removed from the `Skeleton` class, instead they are part of the
    `Annotation` class now: `time`, `edit_position`, `edit_rotation`, `zoom_level`, `task_bounding_box`,
    `user_bounding_boxes`.
  * The following `Annotation` methods were renamed and their arguments changed slightly:
    - `save_volume_annotation` ➜ `export_volume_layer_to_dataset`
    - `temporary_volume_annotation_layer_copy` ➜ `temporary_volume_layer_copy`

### Added
- Added new features to the `Annotation` and `Skeleton` classes. [#602](https://github.com/scalableminds/webknossos-libs/pull/602)
  * The `Skeleton` class has new attributes `description` and `organization_id`.
  * The `Annotation` class has new attributes `username` and `annotation_id`, as well as the following
    attributes that are proxies for the skeleton attributes: `dataset_name`, `scale`, `organization_id`, `description`
  * `Annotation`s can now be initialized from their attributes and an optional skeleton.
  * New methods on `Annotation`: `upload`, `add_volume_layer`, `delete_volume_layer`
  * `Annotation.load()` and `annoation.save()` also works with `.nml` files.
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
- Added `Task.create()` method to create tasks by prodiving a dataset name, location, and rotation. [#605](https://github.com/scalableminds/webknossos-libs/pull/605)


## [0.9.6](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.6) - 2022-02-15
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.9.5...v0.9.6)

### Added
- Added `apply_merger_mode.py` example. [#592](https://github.com/scalableminds/webknossos-libs/pull/592)
- Added support for reading from multiple volume layers in annotations. If an annotation contains multiple volume layers, the layer name has to be provided when reading from a volume layer in an annotation (in `Annotation.save_volume_annotation()` and `Annotation.temporary_volume_annotation_layer_copy()`). Also, added the method `Annotation.get_volume_layer_names()` to see available volume layers. [#588](https://github.com/scalableminds/webknossos-libs/pull/588)

### Changed
- Dataset writes in higher mags do not increase the bounding box if the written data fits in the rounded up box. [#595](https://github.com/scalableminds/webknossos-libs/pull/595)

### Fixed
- Dataset down- & upload: [#595](https://github.com/scalableminds/webknossos-libs/pull/595)
  * Fixed download of higher mags.
  * Bounding box after download is set correctly (was inflated before when downloading higher mags).
  * The returned URL for uploads is corrected, this did not respect `new_dataset_name` before.



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
  * The previous argument `work_on_chunk` is now called `func_per_chunk`.
  * The various `chunk_size` arguments now have to be given in Mag(1). They now have default values.
- Deprecations in `(Mag)View.get_buffered_slice_reader/_writer` [#564](https://github.com/scalableminds/webknossos-libs/pull/564):
  * `(Mag)View.get_buffered_slice_reader`: using the parameters `offset` and `size` is deprecated.
    Please use the parameter relative_bounding_box or absolute_bounding_box (both in Mag(1)) instead.
    The old offset behavior was absolute for `MagView`s and relative for `View`s.
  * `(Mag)View.get_buffered_slice_writer`: using the parameter `offset` is deprecated.
    Please use the parameter relative_offset or absolute_offset (both in Mag(1)) instead.
    The old offset behavior was absolute for `MagView`s and relative for `View`s.


## [0.9.0](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.9.0) - 2022-01-19
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.31...v0.9.0)

### Breaking Changes
- Various changes in View & MagView signatures [#553](https://github.com/scalableminds/webknossos-libs/pull/553):
  - **Breaking Changes**:
    * `MagView.read`: if nothing is supplied and the layer does not start at (0, 0, 0),
      the default behaviour changes from starting at absolute (0, 0, 0) to the layer's bounding box
    * `MagView.write`: if no offset is supplied and the layer does not start at (0, 0, 0),
      the default behaviour changes from starting at absolute (0, 0, 0) to the layer's bounding box
    * `(Mag)View.get_view`: read_only is a keyword-only argument now
    * `MagView.get_bounding_boxes_on_disk()` now returns an iterator yielding bounding boxes in Mag(1)
    * `BoundingBox` cannot have negative topleft or size entries anymore (lifted in v0.9.4).
  - **Deprecations**
    The following usages are marked as deprecated with warnings and will be removed in future releases:
    * Using the `offset` parameter for `read`/`write`/`get_view` in MagView and View is deprecated.
      There are new counterparts `absolute_offset` and `relative_offset` which have to be specified in Mag(1),
      whereas `offset` previously was specified in the Mag of the respective View.
      Also, for `read`/`get_view` only using `size` is deprecated, since it used to refer to the size in the View's Mag.
      Instead, `size` should always be used together with `absolute_offset` or `relative_offset`. Then it is interpreted in Mag(1).
    * The (Mag)View attributes `view.global_offset` and `view.size` are deprecated now, which were in the Mag of the respective View.
      Please use `view.bounding_box` instead, which is in Mag(1).
    * `read_bbox` on the (Mag)View is deprecated as well, please use `read` with the `absolute_bounding_box`or `relative_bounding_box` parameter instead. You'll have to pass the bounding box in Mag(1) then.


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
  * `Dataset.create()` → `Dataset()`
  * `Dataset.get_or_create()` → `Dataset(…, exist_ok=True)`
  * `Dataset()` → `Dataset.open()`
  * `download_dataset()` → `Dataset.download()`
  * `open_annotation()` → `Annotation.load()` for local files, `Annotation.download()` to download from webKnossos
  * `open_nml()` → `Skeleton.load()`
  * `Skeleton.from_path()` → `Skeleton.load()`
  * `Skeleton.write()` → `Skeleton.save()`
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
- Downgraded typing-extensions for better dependency compatibility  [#472](https://github.com/scalableminds/webknossos-libs/pull/472)


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
