import pytest

import webknossos as wk

pytestmark = [pytest.mark.use_proxay]


def test_get_task() -> None:
    team = wk.Team.get_list()[0]
    task_type = wk.TaskType.create(
        name="test_get_task_task_type",
        description="This is a test task type",
        team=team,
    )
    project = wk.Project.create(
        name="test_get_task_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team=team,
        is_blacklisted_from_report=False,
    )

    task = wk.Task.create(
        task_type_id=task_type.task_type_id,
        project_name=project.name,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
    )
    task_by_id = wk.Task.get_by_id(task[0].task_id)
    assert task_by_id == task[0]


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
