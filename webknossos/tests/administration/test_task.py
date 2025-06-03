from collections.abc import Iterator

import pytest

import webknossos as wk

pytestmark = [pytest.mark.use_proxay]

TEST_PROJECT_NAME = "test_project"
TEST_TASK_TYPE_NAME = "test_task_type"


@pytest.fixture(scope="module", autouse=True)
def setup_task_type_and_project() -> Iterator[None]:
    team = wk.Team.get_list()[0]
    task_type = wk.TaskType.create(
        name="test_task_type",
        description="This is a test task type",
        team=team,
        tracing_type="volume",
    )
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


def test_get_task() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.Dataset.open_remote("l4_sample")._dataset_id,
    )
    task_by_id = wk.Task.get_by_id(tasks[0].task_id)
    assert task_by_id == tasks[0]


def test_task_get_project() -> None:
    tasks = wk.Task.create(
        task_type_id=wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME).task_type_id,
        project_name=TEST_PROJECT_NAME,
        needed_experience_domain="testing",
        needed_experience_value=0,
        starting_position=wk.Vec3Int(0, 0, 0),
        dataset_id=wk.Dataset.open_remote("l4_sample")._dataset_id,
    )
    project = tasks[0].get_project()
    assert project.name == TEST_PROJECT_NAME


def test_get_task_type() -> None:
    task_type_by_name = wk.TaskType.get_by_name(TEST_TASK_TYPE_NAME)
    assert task_type_by_name.name == TEST_TASK_TYPE_NAME

    task_type_by_id = wk.TaskType.get_by_id(task_type_by_name.task_type_id)
    assert task_type_by_id == task_type_by_name
