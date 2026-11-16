from collections.abc import Iterator
from typing import Any

import pytest

import webknossos as wk
from webknossos.administration.task import TaskExperience, _api_experience_from_args
from webknossos.client.api_client._serialization import api_client_converter
from webknossos.client.api_client.errors import UnexpectedStatusError
from webknossos.client.api_client.models import (
    ApiExperience,
    ApiTask,
    ApiTaskParameters,
)

pytestmark = [pytest.mark.skip_on_windows]

TEST_PROJECT_NAME = "test_project"
TEST_TASK_TYPE_NAME = "test_task_type"


@pytest.fixture
def setup_task_type_and_project() -> Iterator[None]:
    team = wk.Team.get_list()[0]
    try:
        wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).delete()
    except ValueError:
        pass
    task_type = wk.TaskType.create(
        name="test_task_type",
        description="This is a test task type",
        team=team,
        tracing_type="volume",
    )
    try:
        wk.Project.get_by_name(TEST_PROJECT_NAME).delete()
    except ValueError:
        pass
    project = wk.Project.create(
        name=TEST_PROJECT_NAME,
        priority=1,
        paused=False,
        expected_time=1234,
        team=team,
        is_blacklisted_from_report=False,
    )
    yield
    task_type.delete()
    project.delete()


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_get_task() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.RemoteDataset.open("l4_sample")._dataset_id,
    )
    task_by_id = wk.Task.get_by_id(tasks[0].task_id)
    assert task_by_id == tasks[0]


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_task_get_project() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.RemoteDataset.open("l4_sample")._dataset_id,
    )
    project = tasks[0].get_project()
    assert project.name == TEST_PROJECT_NAME


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_get_task_type() -> None:
    task_type_by_name = wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME)
    assert task_type_by_name.name == TEST_TASK_TYPE_NAME

    task_type_by_id = wk.TaskType.get_by_id(task_type_by_name.task_type_id)
    assert task_type_by_id == task_type_by_name


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_update_task() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.RemoteDataset.open("l4_sample")._dataset_id,
    )
    task = tasks[0]
    updated = task.update(
        remaining_instances=4,
    )
    assert updated.status.pending_instance_count == 4


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_delete_task() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.RemoteDataset.open("l4_sample")._dataset_id,
    )
    task = tasks[0]
    task.delete()
    with pytest.raises(UnexpectedStatusError):
        wk.Task.get_by_id(
            task.task_id
        )  # Should raise KeyError since the task is deleted


@pytest.mark.usefixtures("setup_task_type_and_project")
def test_task_list() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.RemoteDataset.open("l4_sample")._dataset_id,
    )
    task_list = wk.Task.get_list()
    assert len(task_list) > 0
    assert tasks[0] in task_list


def _task_json(needed_experience: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": "6560a1b2c3d4e5f6a7b8c9d0",
        "projectId": "6560a1b2c3d4e5f6a7b8c9d1",
        "datasetId": "6560a1b2c3d4e5f6a7b8c9d2",
        "status": {"pending": 1, "active": 0, "finished": 0},
        "type": {
            "id": "6560a1b2c3d4e5f6a7b8c9d3",
            "summary": "test_task_type",
            "description": "This is a test task type",
            "teamId": "6560a1b2c3d4e5f6a7b8c9d4",
            "teamName": "team_X1",
            "settings": {},
            "tracingType": "volume",
        },
        "neededExperience": needed_experience,
        "boundingBox": None,
        "editPosition": [0, 0, 0],
        "editRotation": [0, 0, 0],
        "script": None,
    }


def test_task_from_json_without_experience() -> None:
    api_task = api_client_converter.structure(_task_json(None), ApiTask)
    assert api_task.needed_experience is None
    task = wk.Task._from_api_task(api_task)
    assert task.experience is None


def test_task_from_json_with_experience() -> None:
    api_task = api_client_converter.structure(
        _task_json({"domain": "testing", "value": 3}), ApiTask
    )
    task = wk.Task._from_api_task(api_task)
    assert task.experience == TaskExperience("testing", 3)


def test_task_parameters_to_json_without_experience() -> None:
    task_parameters = ApiTaskParameters(
        task_type_id="6560a1b2c3d4e5f6a7b8c9d3",
        needed_experience=None,
        pending_instances=1,
        project_name=TEST_PROJECT_NAME,
        script_id=None,
        bounding_box=None,
        dataset_id="6560a1b2c3d4e5f6a7b8c9d2",
        edit_position=(0, 0, 0),
        edit_rotation=(0, 0, 0),
    )
    assert api_client_converter.unstructure(task_parameters)["neededExperience"] is None


@pytest.mark.parametrize(
    ("domain", "value", "expected"),
    [
        (None, None, None),
        ("", None, None),
        ("  ", None, None),
        ("testing", 3, ApiExperience(domain="testing", value=3)),
    ],
)
def test_needed_experience_from_args(
    domain: str | None, value: int | None, expected: ApiExperience | None
) -> None:
    assert _api_experience_from_args(domain, value) == expected


@pytest.mark.parametrize(
    ("domain", "value"),
    [(None, 3), ("", 3), ("testing", None)],
)
def test_needed_experience_from_args_invalid(
    domain: str | None, value: int | None
) -> None:
    with pytest.raises(ValueError, match="needed_experience"):
        _api_experience_from_args(domain, value)


def test_create_task_with_experience_value_but_no_domain() -> None:
    with pytest.raises(ValueError, match="needed_experience_domain"):
        wk.Task.create(
            task_type_id="6560a1b2c3d4e5f6a7b8c9d3",
            project_name=TEST_PROJECT_NAME,
            needed_experience_domain=None,
            needed_experience_value=3,
            starting_position=wk.Vec3Int(0, 0, 0),
            dataset_id="6560a1b2c3d4e5f6a7b8c9d2",
        )
