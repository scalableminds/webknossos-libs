# MeshAttachment Specification

**Current version: 9**

## Overview

A `MeshAttachment` stores multi-resolution (LoD) meshes for all or a subset of segments in a segmentation. It is optimized for fast lookup of a mesh chunk given a `segment_id`, level of detail, and chunk position. Mesh geometry is encoded in [Draco](https://github.com/google/draco) format following the Neuroglancer multi-resolution mesh layout.

## File Format

The artifact is stored as a [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) hierarchy on disk (from version 8 onward).

## Directory Structure

```
<artifact_root>/
  zarr.json          # root group attributes (version, class)
  meshfile/          # single Zarr group containing all mesh data
    zarr.json        # group attributes (version + mesh metadata)
    bucket_offsets/  # array
    buckets/         # array
    neuroglancer/    # array
```

## Root Group Attributes

The root `zarr.json` stores the following under the `voxelytics` key:

| Key                      | Value                   |
|--------------------------|-------------------------|
| `artifact_schema_version`| `9`                     |
| `artifact_class`         | `"MeshfileArtifact"`    |

## Meshfile Group Attributes

The `meshfile/zarr.json` stores mesh-level metadata under the `voxelytics` key, in addition to the version fields above:

| Attribute                       | Type          | Example                      | Description |
|---------------------------------|---------------|------------------------------|-------------|
| `version`                       | `uint32`      | `9`                          | Artifact schema version |
| `global_bounding_box`           | `uint64 (6,)` | `[100,100,100,10,10,10]`     | Bounding box of the full segmentation: `[x, y, z, width, height, depth]` in voxels |
| `mag`                           | `uint32 (3,)` | `[2, 2, 1]`                  | Magnification level the meshes were computed at |
| `mapping_name`                  | `string`      | `"agglomerate_view_30"`      | Name of the agglomerate mapping used; empty string `""` if no mapping was applied |
| `hash_function`                 | `string`      | `"murmurhash3_x64_128"`      | Hash function used for the segment ID hash table |
| `n_buckets`                     | `uint64`      | `n_segments * 1.33`          | Number of hash table buckets (load factor 0.75) |
| `mesh_format`                   | `string`      | `"draco"`                    | Encoding format of individual mesh chunks |
| `n_lod`                         | `int`         | `3`                          | Number of levels of detail |
| `vertex_quantization_bits_per_lod` | `uint8 (n_lod,)` | `[10, 12, 11]`        | Vertex quantization bits per LoD; limits chunk size to `2^bits` voxels per axis |
| `transform`                     | `float32 (3, 4)` | —                         | Affine transform matrix mapping voxel coordinates to physical space (e.g. nanometers) |
| `lod_scale_multiplier`          | `float32`     | `1.0`                        | Global scale multiplier applied to all LoD scales |
| `unit`                          | `string`      | `"nm"`                       | Physical unit of the transformed coordinates |

## Arrays

Let:
- `n_segments` = number of segments with mesh data
- `n_buckets` = `ceil(n_segments / 0.75)` (hash table capacity at load factor 0.75)
- `n_bytes` = total byte size of all manifests and Draco mesh fragments

### `bucket_offsets`

| Property | Value |
|----------|-------|
| Shape    | `(n_buckets + 1,)` |
| Dtype    | `uint64` |

CSR-style offset array into `buckets`. The entries for hash bucket `i` are at `buckets[bucket_offsets[i]:bucket_offsets[i+1]]`. The last entry equals `n_segments`.

### `buckets`

| Property | Value |
|----------|-------|
| Shape    | `(n_segments, 3)` |
| Dtype    | `uint64` |

Each row is `[segment_id, manifest_start, manifest_end]`:

- `segment_id`: the segment this row belongs to
- `manifest_start`: byte offset in `neuroglancer` where the segment's manifest begins
- `manifest_end`: byte offset in `neuroglancer` where the segment's manifest ends

Multiple rows may share the same bucket when hash collisions occur.

### `neuroglancer`

| Property | Value |
|----------|-------|
| Shape    | `(n_bytes,)` |
| Dtype    | `uint8` |

A flat byte array containing, for each segment, all Draco mesh fragments for every LoD followed by the segment's manifest. The layout for one segment is:

```
[Draco fragments LoD 0] [Draco fragments LoD 1] ... [Draco fragments LoD N-1] [Manifest]
```

`manifest_start` (from `buckets`) points to the first byte of the manifest. The Draco fragments for each LoD immediately precede the manifest in ascending LoD order.

## Segment Lookup

To retrieve the mesh for a given `segment_id` and `lod`:

1. Compute `bucket_index = murmurhash3_x64_128(segment_id) % n_buckets`.
2. Read `[start, end] = bucket_offsets[bucket_index:bucket_index+2]`.
3. Read `bucket = buckets[start:end]`.
4. Find the row `r` where `bucket[r, 0] == segment_id`.
5. Read the manifest from `neuroglancer[bucket[r, 1]:bucket[r, 2]]`.
6. Parse the manifest to obtain fragment positions and byte sizes.
7. Compute the byte range of the desired LoD's Draco data relative to `bucket[r, 1]` using the fragment byte sizes in the manifest.

## Manifest Binary Format

Each segment's manifest is a packed binary structure (all multi-byte integers are little-endian):

| Field                    | Type            | Size (bytes)            | Description |
|--------------------------|-----------------|-------------------------|-------------|
| `chunk_shape`            | `float32 (3,)`  | 12                      | Voxel size of the finest-LoD chunk along each axis |
| `grid_origin`            | `float32 (3,)`  | 12                      | Origin of the chunk grid in voxel space |
| `num_lods`               | `uint32`        | 4                       | Number of LoDs stored |
| `lod_scales`             | `float32 (num_lods,)` | `num_lods * 4`    | Scale factor for each LoD relative to LoD 0 |
| `vertex_offsets`         | `float32 (num_lods, 3)` | `num_lods * 12` | Per-LoD vertex origin offset in voxel space |
| `num_fragments_per_lod`  | `uint32 (num_lods,)` | `num_lods * 4`    | Number of fragments at each LoD |
| For each LoD:            |                 |                         | |
| `fragment_positions`     | `uint32 (n_frags, 3)` | `n_frags * 12`  | Grid positions of each fragment; stored column-major (all X, then all Y, then all Z) |
| `fragment_byte_sizes`    | `uint32 (n_frags,)` | `n_frags * 4`    | Byte size of the corresponding Draco binary for each fragment |

The bounding box of fragment at grid position `p` at LoD `l` is:

```
top_left = grid_origin + vertex_offsets[l] + p * chunk_shape * (2^l) * lod_scales[l] * lod_scale_multiplier
size     = chunk_shape * (2^l)
```

## Mesh Encoding

Each fragment is a [Draco](https://github.com/google/draco)-encoded triangle mesh. Vertex coordinates are quantized integers within the fragment's bounding box. To convert to physical space apply the `transform` matrix:

```
physical_coords = matmul(quantized_vertices + bounding_box.topleft, transform)[:, :3]
```

## Chunking

| Array           | Target chunk size | Target shard size |
|-----------------|-------------------|-------------------|
| `bucket_offsets`| 64 KB             | 128 MB            |
| `buckets`       | 128 KB            | 256 MB            |
| `neuroglancer`  | 64 KB             | 512 MB            |

`bucket_offsets` and `buckets` use small chunks to support fine-grained random access during per-segment lookups. `neuroglancer` balances random manifest access against the overhead of many small chunks.

## Artifact Metadata

The following fields are stored in the artifact's voxelytics metadata (not in the Zarr attributes) and configure how meshes were produced:

| Field                     | Type           | Description |
|---------------------------|----------------|-------------|
| `datasource_config`       | `Datasource`   | Configuration of the source segmentation |
| `simplification_factors`  | `List[int]`    | Per-LoD simplification factor relative to the previous LoD (index 0 is relative to the raw marching-cubes output) |
| `max_simplification_errors` | `List[float]`| Maximum geometric error allowed per simplification step |
| `attachment_name`         | `string`       | Name under which this mesh file is registered as an attachment in the WEBKNOSSOS dataset |
