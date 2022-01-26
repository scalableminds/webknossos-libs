import attr

@attr.frozen
class Task:
    id: str
    project_id: str
    dataset_name: str


    @classmethod
    def get_by_id(cls, id: str) -> "Task":
        return Task(id, "dummyProjectId", "dummyDatasetName")