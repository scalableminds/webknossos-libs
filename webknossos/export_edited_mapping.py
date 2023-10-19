import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import DefaultDict, List, Tuple

import numpy as np
import requests
from cluster_tools import Executor, get_executor

import webknossos as wk

logger = logging.getLogger(__name__)


def main():
    # token = "secretSampleUserToken"
    # agglomerate_ids = [1, 63, 25]
    # volume_tracing_id = "9101961c-3678-4204-87d2-2aeeb3af7ea8"
    # tracingstore_uri = "http://localhost:9000"
    # dataset_path = Path("/home/f/scm/code/webknossos/binaryData/sample_organization/test-agglomerate-file")
    # output_dataset_path = Path("/home/f/scm/code/webknossos/binaryData/sample_organization/test-agglomerate-file-with-materialized-mapping")
    # layer_name = "segmentation"
    # output_layer_name = "materialized_mapping"

    token = ""
    agglomerate_ids = [
        23165613,
        27431731,
        27432412,
        27432841,
        23177847,
        27432537,
        19556867,
        27431564,
        27432724,
        27433448,
        27433551,
        27431447,
        25986770,
        22554476,
        24397961,
    ]
    volume_tracing_id = "7904faea-8952-4598-bc46-e910b91c254a"
    tracingstore_uri = "https://webknossos.org"
    dataset_path = Path(
        "/srv/stores/webknossos.org/4fd6473e68256c0a/GJD2_segmentation_v1"
    )
    output_dataset_path = Path(
        "/srv/stores/webknossos.org/4fd6473e68256c0a/GJD2_segmentation_v1_with_materialized_mapping_2023-06-22"
    )
    layer_name = "segmentation"
    output_layer_name = "materialized_mapping_2023-06-22"

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    if output_dataset_path.exists():
        raise FileExistsError(
            f"Output dataset already exists. If re-running after failed run, delete it with  rm -rf {output_dataset_path}"
        )

    mapping = build_mapping(tracingstore_uri, volume_tracing_id, token, agglomerate_ids)
    input_layer, output_layer = set_up_datasets(
        dataset_path, output_dataset_path, layer_name, output_layer_name
    )

    with get_executor("multiprocessing", max_workers=10) as executor:
        apply_mapping(input_layer, output_layer, mapping, executor)
        output_layer.downsample(executor=executor)

    logger.info("Done!")


def set_up_datasets(
    dataset_path: Path,
    output_dataset_path: Path,
    layer_name: str,
    output_layer_name: str,
):
    logger.info(f"Setting up Datasets, output will be at {output_dataset_path}")
    input_dataset = wk.Dataset(dataset_path)
    input_layer = input_dataset.get_layer(layer_name)
    output_dataset = wk.Dataset(
        output_dataset_path, voxel_size=input_dataset.voxel_size
    )
    for layer in input_dataset.layers.values():
        output_dataset.add_symlink_layer(layer, make_relative=True)
    output_layer = output_dataset.add_layer_like(input_layer, output_layer_name)
    return input_layer, output_layer


def build_mapping(
    tracingstore_uri: str,
    volume_tracing_id: str,
    token: str,
    agglomerate_ids: List[int],
) -> DefaultDict[int, int]:
    logger.info(f"Fetching mapping for {len(agglomerate_ids)} selected ids...")
    mapping = defaultdict(zero)

    for agglomerate_id in agglomerate_ids:
        reply = requests.get(
            f"{tracingstore_uri}/tracings/mapping/{volume_tracing_id}/segmentsForAgglomerate?token={token}&agglomerateId={agglomerate_id}"
        )
        reply.raise_for_status()
        reply_parsed = reply.json()
        segment_ids = reply_parsed["segmentIds"]
        for segment_id in segment_ids:
            mapping[segment_id] = agglomerate_id
        present_in_editable_mapping = reply_parsed["agglomerateIdIsPresent"]
        if not present_in_editable_mapping:
            logger.warning(
                f"Agglomerate {agglomerate_id} is not present in editable mapping."
            )
    return mapping


def zero() -> int:
    return 0


def apply_mapping(
    input_layer: wk.Layer,
    output_layer: wk.Layer,
    mapping: DefaultDict[int, int],
    executor: Executor,
):
    input_mag = input_layer.get_finest_mag()
    output_mag = output_layer.add_mag(input_mag.mag.to_tuple(), compress=True)
    input_mag.for_zipped_chunks(
        partial(apply_mapping_for_chunk, mapping=mapping),
        output_mag,
        progress_desc=f"Applying mapping, writing layer {output_layer.name}",
        executor=executor,
    )
    output_layer.bounding_box = input_layer.bounding_box


def apply_mapping_for_chunk(
    args: Tuple[wk.View, wk.View, int], mapping: DefaultDict[int, int]
) -> None:
    (input_view, output_view, _) = args
    cube_data = input_view.read()[0]
    # list comprehension is supposedly faster than for loop,
    # compare https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    # cannot use ids as array index because values might be huge
    # cannot use fastremap with defaultdict (we want to map ids that are not present to zero)
    remapped = np.array(
        [mapping[i] for i in cube_data.flatten()], dtype=cube_data.dtype
    )
    output_view.write(remapped.reshape(cube_data.shape))


if __name__ == "__main__":
    main()
