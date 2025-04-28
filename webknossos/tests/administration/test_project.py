import pytest

import webknossos as wk
from webknossos.client.context import _get_api_client

pytestmark = [pytest.mark.use_proxay]


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


def test_project_update() -> None:
    """Test updating a project."""
    teams = _get_api_client(enforce_auth=True).user_current().teams

    project = wk.Project.create(
        name="test_update_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team_id=teams[0].id,
        is_blacklisted_from_report=False,
    )
    project = project.update(priority=2)
    assert project.priority == 2
    # assert project.paused is True


def test_project_delete() -> None:
    """Test deleting a project."""
    teams = _get_api_client(enforce_auth=True).user_current().teams

    project = wk.Project.create(
        name="test_delete_project",
        priority=1,
        paused=False,
        expected_time=1234,
        team_id=teams[0].id,
        is_blacklisted_from_report=False,
    )
    project.delete()
    with pytest.raises(Exception):
        wk.Project.get_by_name("test_delete_project")
