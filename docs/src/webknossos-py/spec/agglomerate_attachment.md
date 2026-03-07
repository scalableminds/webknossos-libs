# AgglomerateAttachment Specification

**Current version: 4**

An `AgglomerateAttachment` stores the agglomeration graph for a segmentation layer. It maps every segment to an agglomerate and stores, for each agglomerate, its constituent segments, the edges between them, the affinity scores of those edges, and a representative voxel position per segment. There is one attachment directory per mapping (e.g. one per agglomeration threshold).

## File Format

The artifact is stored as a [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) group on disk.

## Directory Structure

```
agglomerate_view_{mapping_id}/
  zarr.json                            # group metadata (version, class)
  segment_to_agglomerate/              # array
  agglomerate_to_segments_offsets/     # array
  agglomerate_to_segments/             # array
  agglomerate_to_edges_offsets/        # array
  agglomerate_to_edges/                # array
  agglomerate_to_affinities/           # array
  agglomerate_to_positions/            # array
```

The `mapping_id` is either an integer (commonly a percentile of the agglomeration score) or a string identifier. The directory is referenced from the layer's `datasource-properties.json` via [`AttachmentsProperties.agglomerates`](datasource_properties.md#attachmentsproperties).

---

## Group Metadata (`zarr.json`)

The group `zarr.json` stores the following attributes under the `voxelytics` key:

| Key | Value |
|---|---|
| `zarr_format` | `3` |
| `node_type` | `"group"` |
| `attributes.voxelytics.artifact_schema_version` | `4` |
| `attributes.voxelytics.artifact_class` | `"AgglomerateViewArtifact"` |

**Example:**

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "voxelytics": {
      "artifact_schema_version": 4,
      "artifact_class": "AgglomerateViewArtifact"
    }
  }
}
```

---

## Notation

Let:

- `n_segments` = total number of segments (segment IDs are 1-based and **dense**: every integer from 1 to `n_segments` must be present; segment 0 is the background)
- `n_agglomerates` = number of real agglomerates (agglomerate 0 is reserved and always empty)
- `n_edges` = total number of edges across all agglomerates

The `segmentation_dtype` is `uint32` when `n_segments < 2^32`, otherwise `uint64`.

---

## Arrays

### `segment_to_agglomerate`

| Property | Value |
|---|---|
| Shape | `(n_segments + 1,)` |
| Dtype | `uint64` |

Maps each segment ID to its agglomerate ID. Index 0 is the background segment and maps to agglomerate 0.

**Example:** `[0, 1, 1, 1, 1, 2, 2, 1]` — segments 1–4 and 7 belong to agglomerate 1, segments 5–6 belong to agglomerate 2.

---

### `agglomerate_to_segments_offsets`

| Property | Value |
|---|---|
| Shape | `(n_agglomerates + 2,)` |
| Dtype | `uint64` |

CSR-style offset array into `agglomerate_to_segments`. The segments belonging to agglomerate `i` are at indices `[offsets[i], offsets[i+1])` in `agglomerate_to_segments`.

Agglomerate 0 is always empty: `offsets[0] == offsets[1] == 0`. The last entry equals `n_segments`.

**Example:** `[0, 0, 5, 7]` — agglomerate 0 is empty, agglomerate 1 has 5 segments (indices 0–4), agglomerate 2 has 2 segments (indices 5–6).

---

### `agglomerate_to_segments`

| Property | Value |
|---|---|
| Shape | `(n_segments,)` |
| Dtype | `segmentation_dtype` |

All segment IDs, grouped by agglomerate. The segments for agglomerate `i` occupy `agglomerate_to_segments[offsets[i]:offsets[i+1]]` and are **sorted in ascending order** within each agglomerate.

**Example:** `[1, 2, 3, 4, 7, 5, 6]` — agglomerate 1 contains segments {1, 2, 3, 4, 7} and agglomerate 2 contains {5, 6}.

---

### `agglomerate_to_edges_offsets`

| Property | Value |
|---|---|
| Shape | `(n_agglomerates + 2,)` |
| Dtype | `uint64` |

CSR-style offset array into `agglomerate_to_edges` and `agglomerate_to_affinities`. The edges for agglomerate `i` are at indices `[offsets[i], offsets[i+1])`. The last entry equals `n_edges`.

Agglomerate 0 is always empty: `offsets[0] == offsets[1] == 0`.

**Example:** `[0, 0, 4, 5]` — agglomerate 1 has 4 edges, agglomerate 2 has 1 edge.

---

### `agglomerate_to_edges`

| Property | Value |
|---|---|
| Shape | `(n_edges, 2)` |
| Dtype | `segmentation_dtype` |

All edges, grouped by agglomerate. Values are **zero-based local node indices** within each agglomerate (i.e. positions within the agglomerate's slice of `agglomerate_to_segments`).

For each edge `(n1, n2)`:
- `n1 < n2`
- Edges within each agglomerate are sorted lexicographically: first by `n1`, then by `n2`.

**Example:** `[[0,1], [0,4], [1,2], [2,3], [0,1]]`

To convert local indices to global segment IDs, index into `agglomerate_to_segments` using the agglomerate's offset from `agglomerate_to_segments_offsets`.

---

### `agglomerate_to_affinities`

| Property | Value |
|---|---|
| Shape | `(n_edges,)` |
| Dtype | `float32` |

Affinity score for each edge, co-indexed with `agglomerate_to_edges`. Higher values indicate stronger evidence for merging.

**Example:** `[124.0, 65.5, 0.0, 250.5, 80.0]`

---

### `agglomerate_to_positions`

| Property | Value |
|---|---|
| Shape | `(n_segments, 3)` |
| Dtype | `int32` |

Representative voxel position `(x, y, z)` for each segment, co-indexed with `agglomerate_to_segments`. The positions for agglomerate `i` occupy `agglomerate_to_positions[offsets[i]:offsets[i+1]]` where `offsets` is `agglomerate_to_segments_offsets`.

---

## Chunking and Sharding

All arrays are written with Zarr v3 sharding (`sharding_indexed` codec). Chunk and shard sizes are derived from the array's shape and dtype to approximate the targets below.

| Array | Target chunk size | Target shard size |
|---|---|---|
| `segment_to_agglomerate` | 256 KB | 1 GB |
| `agglomerate_to_segments` | 256 KB | 1 GB |
| `agglomerate_to_segments_offsets` | 64 KB | 256 MB |
| `agglomerate_to_edges_offsets` | 64 KB | 256 MB |
| `agglomerate_to_edges` | 256 KB | 1 GB |
| `agglomerate_to_affinities` | 256 KB | 1 GB |
| `agglomerate_to_positions` | 256 KB | 1 GB |

The first axis is used as the "row" axis for size calculations; all remaining axes are kept whole in every chunk and shard. The shard shape is always rounded up to the nearest multiple of the chunk shape.

**Codec stack (inner chunks):** `bytes` (little-endian) → `zstd` (level 5, checksum enabled)

**Shard index codecs:** `bytes` (little-endian) → `crc32c`

---

## Invariants

- Segment IDs are 1-based and **dense**: every integer from 1 to `n_segments` must appear as a node.
- Agglomerate 0 is always empty: no segments and no edges.
- Segment IDs within each agglomerate are sorted in ascending order.
- For each edge `(n1, n2)`: `n1 < n2`.
- Edges within each agglomerate are sorted lexicographically `(n1, n2)`.
- Edge node indices are zero-based local indices into the agglomerate's segment list.
- `agglomerate_to_positions` is co-indexed with `agglomerate_to_segments`.
- All offset arrays have shape `(n_agglomerates + 2,)`; the last entry equals the total element count of the corresponding data array.

---

## Worked Example

```
segments:      1, 2, 3, 4, 5, 6, 7
edges:         (1,2), (2,3), (3,4), (5,6), (1,7)

Agglomerate 1: {1, 2, 3, 4, 7}   (connected via all but the 5–6 edge)
Agglomerate 2: {5, 6}

n_agglomerates = 2
n_segments     = 7
n_edges        = 5

segment_to_agglomerate:          [0, 1, 1, 1, 1, 2, 2, 1]   shape (8,)

agglomerate_to_segments_offsets: [0, 0, 5, 7]                shape (4,)
agglomerate_to_segments:         [1, 2, 3, 4, 7, 5, 6]       shape (7,)

agglomerate_to_edges_offsets:    [0, 0, 4, 5]                shape (4,)
agglomerate_to_edges:            [[0,1],[0,4],[1,2],[2,3],    shape (5,2)
                                   [0,1]]
agglomerate_to_affinities:       [124.0, 65.5, 0.0, 250.5,   shape (5,)
                                   80.0]
agglomerate_to_positions:        [[x1,y1,z1], ..., [x7,y7,z7]] shape (7,3)
```

Local edge indices for agglomerate 1 (segments sorted as [1, 2, 3, 4, 7]):

| Global edge | Local (n1, n2) | Sorted? |
|---|---|---|
| 1–2 | (0, 1) | ✓ |
| 1–7 | (0, 4) | ✓ |
| 2–3 | (1, 2) | ✓ |
| 3–4 | (2, 3) | ✓ |

Agglomerate 2 (segments [5, 6]): edge 5–6 → local (0, 1).
