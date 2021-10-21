def test_dataset_usage() -> None:
    from examples.dataset_usage import (
        data_in_mag1,
        data_in_mag1_subset,
        data_in_mag2,
        data_in_mag2_subset,
    )

    assert data_in_mag1.shape == (3, 522, 532, 62)
    assert data_in_mag1_subset.shape == (3, 512, 512, 32)
    assert data_in_mag2.shape == (3, 261, 266, 31)
    assert data_in_mag2_subset.shape == (3, 256, 256, 16)


def test_skeleton_synapse_candidates() -> None:
    from examples.skeleton_synapse_candidates import nml, synapse_parent_group

    assert synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."
