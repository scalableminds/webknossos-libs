import attr


@attr.frozen
class Task:
    id: str
    project_id: str
    dataset_name: str

    @classmethod
    def get_by_id(cls, id: str) -> "Task":  # pylint: disable=redefined-builtin
        from webknossos.client._get_task_info import get_task_info

        return get_task_info(id)
