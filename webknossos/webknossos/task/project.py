from typing import List

import attr

from webknossos.client import User
from webknossos.task import Task

from webknossos.client.context import _get_generated_client

from webknossos.client._generated.api.default import project_info_by_id

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webknossos.client._generated.models.project_info_by_id_response_200 import (
        ProjectInfoResponse200,
    )

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

        """Returns the user specified by the passed id if your token authorizes you to see them."""
        client = _get_generated_client(enforce_auth=True)
        response = project_info_by_id.sync(id, client=client)
        assert response is not None, "Could not fetch project by id."
        return cls._from_generated_response(response)

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        # blocked until generate_client can run against local webKnossos
        pass

    def get_tasks(self) -> List[Task]:
        return []

    def get_owner(self) -> User:
        return User.get_by_id(self.owner_id)

    @classmethod
    def _from_generated_response(cls, response: "ProjectInfoResponse200") -> "Project":
        return Project(
            response.id,
            response.name,
            response.team,
            response.team_name,
            response.owner,
            response.priority,
            response.paused,
            response.expected_time
        )