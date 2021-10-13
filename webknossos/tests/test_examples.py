import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest


@contextmanager
def tmp_cwd() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with TemporaryDirectory() as new_cwd:
        os.chdir(new_cwd)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def test_skeleton_synapse_candidates() -> None:
    from examples.skeleton_synapse_candidates import nml, synapse_parent_group

    assert synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."


@pytest.mark.vcr()
def test_upload_data() -> None:
    with tmp_cwd():
        import examples.upload_image_data


@pytest.mark.vcr()
def test_learned_segmenter() -> None:
    with tmp_cwd():
        import examples.learned_segmenter
