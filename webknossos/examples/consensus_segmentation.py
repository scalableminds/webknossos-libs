import numpy as np
from scipy.ndimage import label

from webknossos import SEGMENTATION_CATEGORY, AnnotationState, Dataset, Project
from webknossos.client._upload_dataset import LayerToLink

OUTPUT_DATASET_NAME = "l4_consensus"
PROJECT_NAME = "l4_segmentation"
DUST_THRESHOLD = 2000
LAYERS_TO_LINK = [LayerToLink("l4_sample", "color")]


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
        masks = [np.isin(vol, seg).astype(np.uint8) for vol, seg in zip(volumes, row)]
        combined = np.sum(np.stack(masks, axis=3), axis=3) > np.ceil(len(volumes) / 2)
        labeled, _ = label(combined)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background
        if len(sizes) > 1:
            # Keep only largest connected component
            largest_label = np.argmax(sizes)
            # This is a sanity check so that correspondences do not overwrite other ones
            if np.all(consensus[labeled == largest_label] == 0):
                consensus[labeled == largest_label] = i + 1
            else:
                print(consensus[labeled == largest_label] == 0)
                raise AssertionError("Overlap in consensus")
    return consensus


if __name__ == "__main__":
    tasks = Project.get_by_name(PROJECT_NAME).get_tasks()
    finished_annotations = []
    for task in tasks:
        for annotation_info in task.get_annotation_infos():
            if annotation_info.state == AnnotationState.FINISHED:
                finished_annotation = annotation_info.download_annotation()
                finished_annotations.append(finished_annotation)
                print("Fetched annotation", finished_annotation.name)

    volumes = [
        a.get_remote_annotation_dataset().get_layer("Volume").get_finest_mag().read()[0]
        for a in finished_annotations
    ]
    print("Downloaded data")

    corr_final = compute_correspondences(volumes)
    print("Computed correspondences", len(corr_final))

    consensus = compute_consensus(volumes, corr_final)
    print("Computed consensus segmentation")

    output_ds = Dataset(OUTPUT_DATASET_NAME, (100,) * 3)
    output_ds.write_layer("consensus", SEGMENTATION_CATEGORY, data=consensus, mag=32)
    print("Wrote out consensus segmentation")

    output_ds.upload(
        new_dataset_name=OUTPUT_DATASET_NAME,
        layers_to_link=LAYERS_TO_LINK,
    )
    print("Uploaded dataset", OUTPUT_DATASET_NAME)
