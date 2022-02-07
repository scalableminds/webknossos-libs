import warnings
from typing import TYPE_CHECKING, List, Union

import attr

from webknossos.administration.user import User
from webknossos.client._generated.api.default import (
    project_info_by_id,
    project_info_by_name,
    task_infos_by_project_id,
)
from webknossos.client.context import _get_generated_client

if TYPE_CHECKING:
    from webknossos.administration import Task
    from webknossos.client._generated.models.project_info_by_id_response_200 import (
        ProjectInfoByIdResponse200,
    )
    from webknossos.client._generated.models.project_info_by_name_response_200 import (
        ProjectInfoByNameResponse200,
    )


@attr.frozen
class Project:
    """Data class containing information about a webKnossos project"""

    project_id: str
    name: str
    team_id: str
    team_name: str
    owner_id: str
    priority: int
    paused: bool
    expected_time: int

    @classmethod
    def get_by_id(
        cls, project_id: str
    ) -> "Project":  # pylint: disable=redefined-builtin
        """Returns the project specified by the passed id if your token authorizes you to see it."""
        client = _get_generated_client(enforce_auth=True)
        response = project_info_by_id.sync(project_id, client=client)
        assert response is not None, "Could not fetch project by id."
        return cls._from_generated_response(response)

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        """Returns the user specified by the passed name if your token authorizes you to see it."""
        client = _get_generated_client(enforce_auth=True)
        response = project_info_by_name.sync(name, client=client)
        assert response is not None, "Could not fetch project by name."
        return cls._from_generated_response(response)

    def get_tasks(self, fetch_all: bool = False) -> List["Task"]:
        """Returns the tasks of this project.
        Note: will fetch only the first 1000 entries by default, warns if that means some are missing.
        set parameter pass fetch_all=True to use pagination to fetch all tasks iteratively with pagination."""

        from webknossos.administration import Task

        PAGINATION_LIMIT = 1000
        pagination_page = 0

        client = _get_generated_client(enforce_auth=True)
        response_raw = task_infos_by_project_id.sync_detailed(
            self.project_id,
            limit=PAGINATION_LIMIT,
            page_number=pagination_page,
            include_total_count=True,
            client=client,
        )
        total_count_raw = response_raw.headers.get("X-Total-Count")
        assert total_count_raw is not None, "X-Total-Count header missing from response"
        total_count = int(total_count_raw)
        response = response_raw.parsed
        assert response is not None, "Could not fetch task infos by project id."
        all_tasks = [Task._from_generated_response(t) for t in response]
        if total_count > PAGINATION_LIMIT:
            if fetch_all:
                while total_count > len(all_tasks):
                    pagination_page += 1
                    response = task_infos_by_project_id.sync(
                        self.project_id,
                        limit=PAGINATION_LIMIT,
                        page_number=pagination_page,
                        include_total_count=False,
                        client=client,
                    )
                    assert (
                        response is not None
                    ), "Could not fetch task infos by project id."
                    new_tasks = [Task._from_generated_response(t) for t in response]
                    all_tasks.extend(new_tasks)

            else:
                warnings.warn(
                    f"Fetched only {PAGINATION_LIMIT} of {total_count} tasks. Pass fetch_all=True to fetch all tasks iteratively (may be slow!)"
                )

        return all_tasks

    def get_owner(self) -> User:
        """Returns the user that is the owner of this task"""
        return User.get_by_id(self.owner_id)

    @classmethod
    def _from_generated_response(
        cls,
        response: Union["ProjectInfoByIdResponse200", "ProjectInfoByNameResponse200"],
    ) -> "Project":
        return cls(
            response.id,
            response.name,
            response.team,
            response.team_name,
            response.owner.id,
            response.priority,
            bool(response.paused),
            response.expected_time,
        )
