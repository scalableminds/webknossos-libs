from math import ceil

import numpy as np
from scipy.ndimage import label

from webknossos import (
    SEGMENTATION_CATEGORY,
    AnnotationState,
    Dataset,
    Project,
    webknossos_context,
)
from webknossos.client._upload_dataset import LayerToLink

OUTPUT_DATASET_NAME = "Y481_ID16A_v9_glomeruli"
PROJECT_NAME = "2025-04-Y481-Glomeruli"
DUST_THRESHOLD = 2000
LAYERS_TO_LINK = [LayerToLink("Y481_ID16A_v9", "color")]


def compute_correspondences(volumes: list[np.ndarray]) -> list[list[list[np.uint32]]]:
    """Computes corresponding segment IDs from multiple segmentation volumes."""
    # First we find out how many voxel have the same segmentation IDs across all volumes
    stacked_volumes = np.stack([v.flatten() for v in volumes], axis=1)
    unique_rows, _, counts = np.unique(
        stacked_volumes, axis=0, return_inverse=True, return_counts=True
    )
    all_corr = np.hstack((unique_rows, counts.reshape(-1, 1)))

    # Then we drop all correspondences for which majority annotated 0
    majority = int(np.ceil(len(volumes) / 2))
    non_majority_zero = np.sum(unique_rows == 0, axis=1) >= majority
    all_corr = all_corr[~non_majority_zero]

    # Now we sort remaining correspondences by number of voxel they share
    sorted_corr = all_corr[np.argsort(-all_corr[:, -1])]

    # And finally use an iterative heuritsic to decide which to keep for correspondence wide voxel majority voting
    corr_final = np.array([sorted_corr[0]])
    alternatives = []

    for candidate in sorted_corr[1:]:
        # Find all segment IDs already in corrFinal (and treat 0 as matching as it does not create new correspondence)
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
                        # In case there is one match only in one row, lets keep correspondence only if it is big 10^3
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

    corr_final_cell = (
        corr_final[:, :-1]
        .reshape(corr_final.shape[0], corr_final.shape[1] - 1, 1)
        .tolist()
    )
    for row in corr_final_cell:
        for i, col in enumerate(row):
            if col == [0]:
                row[i] = []
    for x, y, val in alternatives:
        corr_final_cell[x][y].append(np.uint32(val))
    return corr_final_cell


def compute_consensus(
    volumes: list[np.ndarray], corr: list[list[list[np.uint32]]]
) -> np.ndarray:
    """Compute a consensus segmentation from multiple segmentation volumes based on correspondences."""
    consensus = np.zeros_like(volumes[0], dtype=np.uint32)
    for i, row in enumerate(corr):
        masks = np.stack(
            [np.isin(vol, seg).astype(np.uint8) for vol, seg in zip(volumes, row)],
            axis=3,
        )
        combined = np.sum(masks, axis=3) >= int(ceil(len(volumes) / 2))
        labeled, _ = label(combined)
        sizes = np.bincount(labeled.ravel())
        if len(sizes) > 1:
            sizes[0] = 0  # ignore background
            # Keep only largest connected component
            largest_label = np.argmax(sizes)
            assert largest_label > 0
            # This is a sanity check so that correspondences do not overwrite other ones
            if np.all(consensus[labeled == largest_label] == 0):
                consensus[labeled == largest_label] = i + 1
            else:
                print(consensus[labeled == largest_label])
                print(set(consensus[labeled == largest_label]))
                print(i, row)
                raise AssertionError("Overlap in consensus")
    return consensus


if __name__ == "__main__":
    with webknossos_context(
        url="https://webknossos.crick.ac.uk", token="DLMwYGm92VwhWmreDADj6A"
    ):
        tasks = Project.get_by_name(PROJECT_NAME).get_tasks()
        finished_annotations = []
        for task in tasks:
            for annotation_info in task.get_annotation_infos():
                if annotation_info.state == AnnotationState.FINISHED:
                    finished_annotation = annotation_info.download_annotation()
                    finished_annotations.append(finished_annotation)
                    print("Fetched annotation", finished_annotation.name)

        volumes = [
            a.get_remote_annotation_dataset()
            .get_layer("Volume")
            .get_finest_mag()
            .read()[0]
            for a in finished_annotations
        ]
        print("Downloaded data")

    volumes[2][volumes[2] == 49] = 44
    volumes[1][volumes[2] == 11] = 63
    volumes[1][volumes[2] == 55] = 44

    corr_final = compute_correspondences(volumes)
    print("Computed correspondences", len(corr_final))

    for row in corr_final:
        print(row)

    consensus = compute_consensus(volumes, corr_final)
    print("Computed consensus segmentation")

    print(np.stack(volumes, axis=0)[:, 2957 // 32, 4381 // 32, 1374 // 32])

    assert consensus[2957 // 32, 4381 // 32, 1374 // 32] == 2
    assert consensus[3466 // 32, 2146 // 32, 1053 // 32] != 0

    assert consensus[3053 // 32, 6531 // 32, 2063 // 32] != 0
    assert consensus[3634 // 32, 1891 // 32, 1361 // 32] != 0
    assert consensus[6674 // 32, 5430 // 32, 1435 // 32] != 0
    assert consensus[6323 // 32, 3568 // 32, 1525 // 32] != 0
    assert consensus[6687 // 32, 5226 // 32, 1621 // 32] != 0

    consensus[volumes[2] == 68] = len(corr_final) + 2

    output_ds = Dataset(OUTPUT_DATASET_NAME, (100,) * 3)
    output_ds.write_layer("consensus", SEGMENTATION_CATEGORY, data=consensus, mag=32)
    print("Wrote out consensus segmentation")

    uploaded_ds = output_ds.upload(
        new_dataset_name=OUTPUT_DATASET_NAME,
        layers_to_link=LAYERS_TO_LINK,
    )
    print("Uploaded dataset", OUTPUT_DATASET_NAME, uploaded_ds.url)
