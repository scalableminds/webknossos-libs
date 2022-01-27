from typing import List

import attr

from webknossos.client import User
from webknossos.task import Task


@attr.frozen
class Project:
    id: str
    name: str
    team_id: str
    team_name: str
    owner_id: str
    priority: int
    paused: bool
    expected_time: int

    @classmethod
    def get_by_id(cls, id: str) -> "Project":  # pylint: disable=redefined-builtin
        pass

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        pass

    def get_tasks(self) -> List[Task]:
        return []

    def get_owner(self) -> User:
        return User.get_by_id(self.owner_id)
