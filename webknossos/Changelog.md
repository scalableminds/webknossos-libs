# Change Log

All notable changes to the webknossos python library are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/) `MAJOR.MINOR.PATCH`.
For upgrade instructions, please check the respective *Breaking Changes* sections.

## Unreleased
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.25...HEAD)

### Breaking Changes

### Added

### Changed

## [v0.8.25](https://github.com/scalableminds/webknossos-libs/releases/tag/v0.8.25) - 2021-12-07
[Commits](https://github.com/scalableminds/webknossos-libs/compare/v0.8.24...v0.8.25)

### Added
- Added support to download datasets from external datastores, which is the case for webknossos.org.  [#497](https://github.com/scalableminds/webknossos-libs/pull/497)

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
