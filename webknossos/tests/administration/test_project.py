import pytest

import webknossos as wk

pytestmark = [pytest.mark.use_proxay]


def test_get_project() -> None:
    teams = wk.Team.get_list()

    project = wk.Project.create(
        name="test_get_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0],
        is_blacklisted_from_report=False,
    )
    project_by_name = wk.Project.get_by_name(project.name)
    assert project_by_name == project
    project_by_id = wk.Project.get_by_id(project.project_id)
    assert project_by_id == project


def test_project_create() -> None:
    """Test creating a project."""

    teams = wk.Team.get_list()

    project = wk.Project.create(
        name="test_create_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0],
        is_blacklisted_from_report=False,
    )
    assert project.name == "test_create_project"
    assert project.priority == 1
    assert project.paused is False
    assert project.expected_time == 1234
    assert project.team_id == teams[0].id
    assert project.is_blacklisted_from_report is False
    assert project.owner_id == wk.User.get_current_user().user_id


def test_project_update() -> None:
    """Test updating a project."""
    teams = wk.Team.get_list()

    project = wk.Project.create(
        name="test_update_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0],
        is_blacklisted_from_report=False,
    )
    project = project.update(
        priority=2,
        expected_time=5678,
        is_blacklisted_from_report=True,
    )
    assert project.priority == 2
    assert project.expected_time == 5678
    assert project.is_blacklisted_from_report is True


def test_project_delete() -> None:
    """Test deleting a project."""
    teams = wk.Team.get_list()

    project = wk.Project.create(
        name="test_delete_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0],
        is_blacklisted_from_report=False,
    )
    project.delete()
    with pytest.raises(Exception):
        wk.Project.get_by_name("test_delete_project")


def test_project_get_owner() -> None:
    """Test getting the owner of a project."""
    teams = wk.Team.get_list()

    project = wk.Project.create(
        name="test_get_owner_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0],
        is_blacklisted_from_report=False,
    )
    owner = project.get_owner()
    assert owner.user_id == wk.User.get_current_user().user_id


def test_project_get_tasks() -> None:
    """Test getting the task of a project."""
    teams = wk.Team.get_list()
    ds_id = wk.Dataset.open_remote("l4_sample")._dataset_id

    project = wk.Project.create(
        name="test_get_task_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=teams[0].id,
        is_blacklisted_from_report=False,
    )
    task_type = wk.TaskType.create(
        name="test_task_type",
        description="test_task_type_description",
        team=teams[0].name,
        tracing_type="volume",
    )
    for i in range(3):
        wk.Task.create(
            task_type_id=task_type.task_type_id,
            project_name=project.name,
            needed_experience_domain="test_domain",
            needed_experience_value=3,
            starting_position=wk.Vec3Int(0, 0, 0),
            dataset_id=ds_id,
        )

    assert len(project.get_tasks()) == 3
