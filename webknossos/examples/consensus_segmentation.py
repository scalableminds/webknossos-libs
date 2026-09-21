import re
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import webknossos as wk
from webknossos.client._resolve_short_link import resolve_short_link

# Each entry may be a regular annotation URL or a CompoundTask URL.
# For CompoundTask URLs, all finished annotations of the task are used.
ANNOTATION_URLS = [
    "https://webknossos.org/annotations/CompoundTask/<task_id_1>",
    "https://webknossos.org/annotations/CompoundTask/<task_id_2>",
    "https://webknossos.org/annotations/CompoundTask/<task_id_3>",
]
OUTPUT_DATASET_PATH = Path("consensus_segmentation")
OUTPUT_DATASET_NAME = "consensus_segmentation"
COLOR_LAYER_NAME = "color"
# Correspondences with fewer voxels than this are considered dust
DUST_THRESHOLD = 2000

_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://[^/]+)/annotations/((?P<annotation_type>[A-Za-z]+)/)?(?P<id>[0-9A-Fa-f]+)"
)


def compute_correspondences(volumes: list[np.ndarray]) -> list[list[list[int]]]:
    """Computes corresponding segment IDs from multiple segmentation volumes."""
    # First we find out how many voxel have the same segmentation IDs across all volumes
    stacked_volumes = np.stack([v.flatten() for v in volumes], axis=1)
    unique_rows, counts = np.unique(stacked_volumes, axis=0, return_counts=True)
    all_corr = np.hstack((unique_rows, counts.reshape(-1, 1)))

    # Then we drop all correspondences for which the majority annotated 0
    majority = int(np.ceil(len(volumes) / 2))
    majority_zero = np.sum(unique_rows == 0, axis=1) >= majority
    all_corr = all_corr[~majority_zero]

    # Now we sort remaining correspondences by number of voxel they share
    sorted_corr = all_corr[np.argsort(-all_corr[:, -1])]

    # And finally use an iterative heuristic to decide which to keep for correspondence-wide voxel majority voting
    corr_final = np.array([sorted_corr[0]])
    alternatives = []

    for candidate in sorted_corr[1:]:
        # Find all segment IDs already in corr_final (and treat 0 as matching as it does not create new correspondence)
        matched = np.equal(corr_final[:, :-1], candidate[:-1])
        zero_matched = (corr_final[:, :-1] == 0) | (candidate[:-1] == 0)
        matched[zero_matched] = False

        if matched.any():
            if np.any(np.all(matched | zero_matched, axis=1)):
                # We already have this correspondence (somebody voted zero for some region of this corr)
                continue
            if np.sum(~matched.any(axis=0)) > majority:
                # We have a correspondence that is not shared between all annotators
                corr_final = np.append(corr_final, [candidate], axis=0)
            elif candidate[-1] > DUST_THRESHOLD:
                rmatches = matched[np.any(matched, axis=1)]
                if rmatches.shape[0] == 1 and np.sum(rmatches) in [1, 2]:
                    if np.sum(rmatches) == 1:
                        # In case there is one match only in one row, keep the correspondence only if it is big
                        corr_final = np.append(corr_final, [candidate], axis=0)
                    elif np.sum(rmatches) == 2:
                        # In case there are two matches only in one row, we will let the non-matched one vote there
                        inv_matches = ~matched[np.any(matched, axis=1)]
                        x, y = np.where(inv_matches)
                        alternatives.append((x[0], y[0], candidate[y[0]]))
                else:
                    # Now we have a real merger, i.e. non-zero matches in at least two rows
                    print("Merger dropped:", candidate, rmatches)
        else:
            # No overlap with correspondences so far, so we add it
            corr_final = np.append(corr_final, [candidate], axis=0)

    corr_final_cell = [
        [[] if seg_id == 0 else [int(seg_id)] for seg_id in row[:-1]]
        for row in corr_final
    ]
    for x, y, val in alternatives:
        corr_final_cell[x][y].append(int(val))
    return corr_final_cell


def compute_consensus(
    volumes: list[np.ndarray], corr: list[list[list[int]]]
) -> np.ndarray:
    """Compute a consensus segmentation from multiple segmentation volumes based on correspondences."""
    consensus = np.zeros_like(volumes[0], dtype=np.uint32)
    for i, row in enumerate(corr):
        masks = [np.isin(vol, seg).astype(np.uint8) for vol, seg in zip(volumes, row)]
        combined = np.sum(np.stack(masks, axis=-1), axis=-1) > len(volumes) / 2
        # Correspondences are ordered by size, so earlier ones take precedence in case of overlaps
        contested = combined & (consensus != 0)
        if contested.any():
            print(f"Correspondence {i + 1}: {contested.sum()} voxels already assigned")
        combined &= ~contested
        labeled, _ = label(combined)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background
        if len(sizes) > 1:
            # Keep only largest connected component
            largest_label = np.argmax(sizes)
            consensus[labeled == largest_label] = i + 1
    return consensus


def fetch_annotations(urls: list[str]) -> list[wk.Annotation]:
    annotations = []
    for url in urls:
        match = _URL_REGEX.match(resolve_short_link(url))
        assert match is not None, f"Could not parse annotation url {url}"
        if match.group("annotation_type") == "CompoundTask":
            task = wk.Task.get_by_id(match.group("id"))
            for annotation_info in task.get_annotation_infos():
                if annotation_info.state == wk.AnnotationState.FINISHED:
                    annotations.append(annotation_info.download_annotation())
                    print(
                        "Fetched annotation",
                        annotations[-1].name,
                        "of task",
                        task.task_id,
                    )
        else:
            annotations.append(wk.Annotation.download(match.group("id")))
            print("Fetched annotation", annotations[-1].name)
    return annotations


def main() -> None:
    annotations = fetch_annotations(ANNOTATION_URLS)
    assert len(annotations) > 1, "Need at least two annotations for a consensus"

    # All annotations are read within the same bounding box, which is the task bounding box if present
    bounding_box = annotations[0].task_bounding_box
    assert bounding_box is not None, "Annotations must belong to a task"
    for a in annotations:
        assert a.task_bounding_box == bounding_box, (
            "All annotations must share the same task bounding box"
        )
    print("Bounding box", bounding_box)

    volumes = []
    mag = None
    for a in annotations:
        with a.temporary_volume_layer_copy() as layer:
            mag_view = layer.get_finest_mag()
            assert mag is None or mag == mag_view.mag, (
                "Annotations must share the same mag"
            )
            mag = mag_view.mag
            bounding_box = bounding_box.align_with_mag(mag, ceil=True)
            volumes.append(mag_view.read(absolute_bounding_box=bounding_box)[0])
    assert mag is not None
    print("Read data of", len(volumes), "annotations at", mag)

    corr_final = compute_correspondences(volumes)
    print("Computed correspondences", len(corr_final))

    consensus = compute_consensus(volumes, corr_final)
    print(
        "Computed consensus segmentation with",
        len(np.unique(consensus)) - 1,
        "segments",
    )

    output_ds = wk.Dataset(
        OUTPUT_DATASET_PATH,
        voxel_size=annotations[0].voxel_size,
        name=OUTPUT_DATASET_NAME,
        exist_ok=True,
    )
    output_ds.write_layer(
        "consensus",
        wk.SEGMENTATION_CATEGORY,
        data=consensus,
        mag=mag,
        absolute_offset=bounding_box.topleft,
    )
    print("Wrote out consensus segmentation")

    color_layer = annotations[0].get_remote_base_dataset().get_layer(COLOR_LAYER_NAME)
    remote_ds = output_ds.upload(
        new_dataset_name=OUTPUT_DATASET_NAME,
        layers_to_link=[wk.LayerToLink.from_remote_layer(color_layer)],
    )
    print("Uploaded dataset", remote_ds.url)


if __name__ == "__main__":
    main()
