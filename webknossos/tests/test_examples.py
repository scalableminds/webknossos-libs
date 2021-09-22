def test_skeleton_synapse_candidates() -> None:
    from examples.skeleton_synapse_candidates import nml, synapse_parent_group

    assert synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."
