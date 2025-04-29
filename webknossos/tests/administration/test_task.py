import pytest

import webknossos as wk
from webknossos.client.context import _get_api_client

pytestmark = [pytest.mark.use_proxay]


def test_get_task() -> None:
    teams = _get_api_client(enforce_auth=True).user_current().teams

    project = wk.Task.create(
        task_type_id: str,
        project_name: str,
        needed_experience_domain: str,
        needed_experience_value: int,
        starting_position: Vec3Int,
    )
    project_by_name = wk.Project.get_by_name(project.name)
    assert project_by_name == project
    project_by_id = wk.Project.get_by_id(project.project_id)
    assert project_by_id == project


def test_project_create() -> None:
    """Test creating a project."""

    teams = _get_api_client(enforce_auth=True).user_current().teams

    project = wk.Project.create(
        name="test_create_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team_id=teams[0].id,
        is_blacklisted_from_report=False,
    )
    assert project.name == "test_create_project"
    assert project.priority == 1
    assert project.paused is False
    assert project.expected_time == 1234
    assert project.team_id == teams[0].id

