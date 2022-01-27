from webknossos.administration.task import Task
from webknossos.client._generated.api.default import task_info
from webknossos.client.context import _get_generated_client


def get_task_info(id: str) -> Task:  # pylint: disable=redefined-builtin
    client = _get_generated_client()
    response = task_info.sync(id=id, client=client)
    assert response is not None
    return Task(response.id, response.project_name, response.data_set)
