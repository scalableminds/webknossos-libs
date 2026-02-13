# datasource-properties.json Specification

This document specifies the format of `datasource-properties.json`, the metadata file that describes a WEBKNOSSOS dataset. It is located at the root of each dataset directory.

## Top-Level Object: `DatasetProperties`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | [Id](#id) | Yes | | Legacy identifier for the dataset. WEBKNOSSOS derives the actual name and organization from the dataset path, not from this field. |
| `scale` | [VoxelSize](#voxelsize) or `[number, number, number]` | Yes | | Physical size of one voxel. Can be specified as a `VoxelSize` object or as a legacy 3-element array (interpreted as nanometers). |
| `dataLayers` | Array of [LayerProperties](#layerproperties) | Yes | | The layers contained in this dataset. |
| `version` | `integer` | No | `1` | Schema version number. Defaults to `1` if omitted. All new files should include this field and use `1`.  |
| `defaultViewConfiguration` | [DatasetViewConfiguration](#datasetviewconfiguration) | No | `null` | Default view settings applied when opening the dataset in WEBKNOSSOS. |

### Id

A legacy object. WEBKNOSSOS ignores these values and derives them from the dataset path.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | Yes | Dataset directory name. |
| `team` | `string` | Yes | Organization identifier. |

### VoxelSize

Describes the physical size of a single voxel.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `factor` | `[number, number, number]` | Yes | | Voxel dimensions as `[x, y, z]`. |
| `unit` | [LengthUnit](#lengthunit) | No | `"nanometer"` | Unit of the voxel size factor. |

**Legacy format:** `scale` may also be a plain `[number, number, number]` array. In that case the unit is implicitly `"nanometer"`.

### LengthUnit

A string enum. Common values:

| Value | Abbreviation |
|---|---|
| `"nanometer"` | nm |
| `"micrometer"` | um |
| `"millimeter"` | mm |
| `"centimeter"` | cm |
| `"meter"` | m |
| `"angstrom"` | A |

Additional supported values: `"yoctometer"`, `"zeptometer"`, `"attometer"`, `"femtometer"`, `"picometer"`, `"decimeter"`, `"hectometer"`, `"kilometer"`, `"megameter"`, `"gigameter"`, `"terameter"`, `"petameter"`, `"exameter"`, `"zettameter"`, `"yottameter"`, `"inch"`, `"foot"`, `"yard"`, `"mile"`, `"parsec"`.

---

## LayerProperties

Each entry in `dataLayers` describes one layer. The schema varies slightly based on `category` and `dataFormat`.

### Common Fields (all layers)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `string` | Yes | | Layer name. Must be unique within the dataset. Corresponds to the directory name on disk. |
| `category` | `"color"` or `"segmentation"` | Yes | | Layer type. Determines which additional fields are available. |
| `boundingBox` | [BoundingBox](#boundingbox) | Yes | | Spatial extent of the layer at mag 1. |
| `elementClass` | [ElementClass](#elementclass) | Yes | | Data type of each voxel element. |
| `dataFormat` | [DataFormat](#dataformat) | Yes | | On-disk storage format. |
| `numChannels` | `integer` | No | `null` | Number of channels. Relevant for multi-channel color layers (e.g. `3` for RGB stored as `uint24`). |
| `defaultViewConfiguration` | [LayerViewConfiguration](#layerviewconfiguration) | No | `null` | Default view settings for this layer. |

### Magnification Fields

All layers use `mags` (array of [MagViewProperties](#magviewproperties)) to describe their magnification levels.

Deprecated: WKW layers may also use `wkwResolutions` (array of [WkwMagViewProperties](#wkwmagviewproperties)). This is a legacy format; prefer `mags` for all data formats.

### Segmentation Layer Fields

When `category` is `"segmentation"`, the following additional fields are available:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `largestSegmentId` | `integer` | No | `null` | Highest segment ID present in the layer. Used by WEBKNOSSOS for ID management. |
| `mappings` | `string[]` | No | `[]` | List of available ID mapping names (e.g. for agglomerate mappings). |
| `attachments` | [AttachmentsProperties](#attachmentsproperties) | No | `null` | References to auxiliary data files associated with this segmentation layer. Omitted if all sub-fields are empty. |

### N-Dimensional / 4D Layer Fields

For layers with additional axes beyond x, y, z (e.g. time):

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `additionalAxes` | Array of [AdditionalAxis](#additionalaxis) | No | `null` | Extra dimensions beyond x, y, z. |

#### AdditionalAxis

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | Yes | Axis name (e.g. `"t"` for time). |
| `bounds` | `[integer, integer]` | Yes | Inclusive lower and exclusive upper bound for this axis. |
| `index` | `integer` | Yes | Position of this axis in the underlying array's dimension order. |

---

## BoundingBox

Describes the spatial extent of a layer in voxel coordinates at magnification 1.

| Field | Type | Required | Description |
|---|---|---|---|
| `topLeft` | `[integer, integer, integer]` | Yes | Origin corner as `[x, y, z]`. |
| `width` | `integer` | Yes | Extent along the x axis. |
| `height` | `integer` | Yes | Extent along the y axis. |
| `depth` | `integer` | Yes | Extent along the z axis. |

---

## MagViewProperties

Describes a single magnification level for non-WKW formats.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `mag` | `[integer, integer, integer]` | Yes | | Downsampling factor as `[x, y, z]` (e.g. `[2, 2, 2]` for 2x downsampling). |
| `path` | `string` | No | `null` | Relative path to the mag data directory (e.g. `"./color/1"`). |
| `cubeLength` | `integer` | No | `null` | Shard/chunk cube length. Deprecated. |
| `axisOrder` | `object` | Conditional | `null` | Maps axis names to dimension indices in the underlying array (e.g. `{"c": 0, "x": 4, "y": 3, "z": 2}`). Required when the `c` (optional), `x`, `y`, `z` axes are not the last axes in that order. Replicated on every mag but must be identical across all mags. |

## WkwMagViewProperties (Deprecated)

Deprecated: Use [MagViewProperties](#magviewproperties) with `mags` instead for all data formats.

Legacy format used within `wkwResolutions` for WKW layers. Uses `resolution` instead of `mag`.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `resolution` | `[integer, integer, integer]` | Yes | | Downsampling factor as `[x, y, z]`. Same semantics as `mag` in `MagViewProperties`. |
| `cubeLength` | `integer` | No | `null` | WKW file cube edge length in voxels. |

---

## ElementClass

A string indicating the data type of voxel values. Not all element classes are valid for every layer category.

| Value | Description | Color Layers | Segmentation Layers |
|---|---|---|---|
| `"uint8"` | Unsigned 8-bit integer | Yes | Yes |
| `"uint16"` | Unsigned 16-bit integer | Yes | Yes |
| `"uint24"` | Unsigned 24-bit integer (3-channel RGB) | Yes | No |
| `"uint32"` | Unsigned 32-bit integer | Yes | Yes |
| `"uint64"` | Unsigned 64-bit integer | No | Yes (values limited to 2^53 - 1) |
| `"int8"` | Signed 8-bit integer | Yes | Yes |
| `"int16"` | Signed 16-bit integer | Yes | Yes |
| `"int32"` | Signed 32-bit integer | Yes | Yes |
| `"int64"` | Signed 64-bit integer | No | Yes |
| `"float"` | 32-bit floating point | Yes | No |
| `"double"` | 64-bit floating point | No | No |

**Notes:**
- `uint24` is only meaningful for color layers where 3 channels are packed into one element.
- `uint64` segmentation values are internally handled as JavaScript numbers, limiting the usable range to 2^53 - 1.
- `double` is not supported by WEBKNOSSOS for any layer type.
- Signed integers for segmentation layers may cause issues with negative segment IDs in some workflows.

---

## DataFormat

A string indicating the on-disk storage format.

| Value | Description |
|---|---|
| `"zarr3"` | Zarr v3 format. |
| `"zarr"` | Zarr v2 format. |
| `"wkw"` | WEBKNOSSOS Wrap format. |
| `"n5"` | N5 format. |
| `"neuroglancerPrecomputed"` | Neuroglancer Precomputed format. |

---

## DatasetViewConfiguration

Optional defaults for how the dataset is displayed in WEBKNOSSOS.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `fourBit` | `boolean` | No | `null` | Enable 4-bit rendering mode. |
| `interpolation` | `boolean` | No | `null` | Enable interpolation between voxels. |
| `renderMissingDataBlack` | `boolean` | No | `null` | Render missing data as black instead of transparent. |
| `loadingStrategy` | `string` | No | `null` | Data loading strategy. |
| `segmentationPatternOpacity` | `integer` | No | `null` | Opacity of the segmentation hatch pattern. |
| `zoom` | `number` | No | `null` | Initial zoom level. |
| `position` | `[integer, integer, integer]` | No | `null` | Initial camera position as `[x, y, z]`. |
| `rotation` | `[integer, integer, integer]` | No | `null` | Initial camera rotation. |

All fields are optional. Only non-null values are serialized.

---

## LayerViewConfiguration

Optional defaults for how a single layer is displayed in WEBKNOSSOS.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `color` | `[integer, integer, integer]` | No | `null` | RGB color from 0 to 255. WEBKNOSSOS default: `[255, 255, 255]`. |
| `alpha` | `number` | No | `null` | Opacity from 0 to 100. WEBKNOSSOS default: `100` (or `20` for segmentation layers). |
| `intensityRange` | `[number, number]` | No | `null` | Min and max data value range for thresholding. Default: full value range. |
| `min` | `number` | No | `null` | Minimum data value that might be encountered. Restricts the histogram. |
| `max` | `number` | No | `null` | Maximum data value that might be encountered. Restricts the histogram. |
| `isDisabled` | `boolean` | No | `null` | Disable the layer. Default: `false`. |
| `isInverted` | `boolean` | No | `null` | Invert the layer. Default: `false`. |
| `isInEditMode` | `boolean` | No | `null` | Enable histogram edit mode. Default: `false`. |
| `mapping` | `object` | No | `null` | Enable an ID mapping for a segmentation layer. Expected format: `{"name": "<mapping_name>", "type": "HDF5"}`. |

All fields are optional. Only non-null values are serialized.

---

## AttachmentsProperties

References to auxiliary data files for a segmentation layer.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `meshes` | Array of [AttachmentProperties](#attachmentproperties) | No | `null` | Precomputed mesh files. |
| `agglomerates` | Array of [AttachmentProperties](#attachmentproperties) | No | `null` | Agglomerate mapping files. |
| `segmentIndex` | [AttachmentProperties](#attachmentproperties) | No | `null` | Segment index file. |
| `cumsum` | [AttachmentProperties](#attachmentproperties) | No | `null` | Cumulative sum file. |
| `connectomes` | Array of [AttachmentProperties](#attachmentproperties) | No | `null` | Connectome files. |

The entire `attachments` object is omitted from the JSON if all sub-fields are empty or null.

### AttachmentProperties

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | Yes | Display name of the attachment. |
| `path` | `string` | Yes | Relative path to the attachment data. |
| `dataFormat` | `"zarr3"`, `"hdf5"`, or `"json"` | Yes | Storage format of the attachment. |

---

## Serialization Notes

- All field names in the JSON use **camelCase** (e.g. `dataLayers`, `elementClass`, `boundingBox`).
- Optional fields with `null` / default values are **omitted** from the serialized JSON.
- The `additionalAxes` field (for nD layers) is serialized at the layer level, not nested inside `boundingBox`.
- The `axisOrder` and `channelIndex` fields (internal to the bounding box representation) are serialized on each mag entry rather than on the layer or bounding box.

---

## Examples

### Minimal WKW dataset

```json
{
  "version": 1,
  "id": { "name": "my_dataset", "team": "" },
  "scale": [11.24, 11.24, 28.0],
  "dataLayers": [
    {
      "name": "color",
      "category": "color",
      "boundingBox": { "topLeft": [0, 0, 0], "width": 1024, "height": 1024, "depth": 512 },
      "elementClass": "uint8",
      "dataFormat": "wkw",
      "mags": [
        { "mag": [1, 1, 1], "path": "./color/1" },
        { "mag": [2, 2, 2], "path": "./color/2" }
      ]
    }
  ]
}
```

### Zarr3 dataset with view configuration

```json
{
  "version": 1,
  "id": { "name": "my_zarr3_dataset", "team": "" },
  "scale": { "factor": [1.0, 1.0, 1.0], "unit": "micrometer" },
  "dataLayers": [
    {
      "name": "color",
      "category": "color",
      "boundingBox": { "topLeft": [0, 0, 0], "width": 256, "height": 256, "depth": 256 },
      "elementClass": "uint8",
      "dataFormat": "zarr3",
      "numChannels": 3,
      "mags": [
        { "mag": [1, 1, 1], "path": "./color/1" }
      ],
      "defaultViewConfiguration": { "color": [255, 0, 0] }
    }
  ],
  "defaultViewConfiguration": { "position": [128, 128, 128] }
}
```

### 4D dataset with additional axes

```json
{
  "version": 1,
  "id": { "name": "4d_timeseries", "team": "" },
  "scale": [10.0, 10.0, 10.0],
  "dataLayers": [
    {
      "name": "color",
      "category": "color",
      "boundingBox": { "topLeft": [0, 0, 0], "width": 439, "height": 167, "depth": 5 },
      "elementClass": "int8",
      "dataFormat": "zarr3",
      "numChannels": 1,
      "mags": [
        { "mag": [1, 1, 1], "axisOrder": { "c": 0, "x": 4, "y": 3, "z": 2 } },
        { "mag": [2, 2, 2], "axisOrder": { "c": 0, "x": 4, "y": 3, "z": 2 } }
      ],
      "additionalAxes": [
        { "name": "t", "bounds": [0, 7], "index": 1 }
      ]
    }
  ]
}
```
