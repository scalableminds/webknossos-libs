import warnings
from typing import TYPE_CHECKING, List, Optional

import attr

from ..client.api_client.models import ApiProject
from ..client.context import _get_api_client
from .user import User

if TYPE_CHECKING:
    from .task import Task


@attr.frozen
class Project:
    """Data class containing information about a WEBKNOSSOS project"""

    project_id: str
    name: str
    team_id: str
    team_name: str
    owner_id: Optional[str]  # None in case you have no read access on the owner
    priority: int
    paused: bool
    expected_time: Optional[int]

    @classmethod
    def get_by_id(cls, project_id: str) -> "Project":
        """Returns the project specified by the passed id if your token authorizes you to see it."""
        api_project = _get_api_client(enforce_auth=True).project_info_by_id(project_id)
        return cls._from_api_project(api_project)

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        """Returns the user specified by the passed name if your token authorizes you to see it."""
        api_project = _get_api_client(enforce_auth=True).project_info_by_name(name)
        return cls._from_api_project(api_project)

    def get_tasks(self, fetch_all: bool = False) -> List["Task"]:
        """Returns the tasks of this project.
        Note: will fetch only the first 1000 entries by default, warns if that means some are missing.
        set parameter pass fetch_all=True to use pagination to fetch all tasks iteratively with pagination.
        """

        from .task import Task

        PAGINATION_LIMIT = 1000
        pagination_page = 0

        client = _get_api_client(enforce_auth=True)
        api_tasks_batch, total_count = client.task_infos_by_project_id_paginated(
            self.project_id, limit=PAGINATION_LIMIT, page_number=pagination_page
        )
        all_tasks = [Task._from_api_task(t) for t in api_tasks_batch]
        if total_count > PAGINATION_LIMIT:
            if fetch_all:
                while total_count > len(all_tasks):
                    pagination_page += 1
                    api_tasks_batch, _ = client.task_infos_by_project_id_paginated(
                        self.project_id,
                        limit=PAGINATION_LIMIT,
                        page_number=pagination_page,
                    )
                    new_tasks = [Task._from_api_task(t) for t in api_tasks_batch]
                    all_tasks.extend(new_tasks)

            else:
                warnings.warn(
                    f"[INFO] Fetched only {PAGINATION_LIMIT} of {total_count} tasks. Pass fetch_all=True to fetch all tasks iteratively (may be slow!)"
                )

        return all_tasks

    def get_owner(self) -> User:
        """Returns the user that is the owner of this task"""
        assert (
            self.owner_id is not None
        ), "Project owner is None, you may not have enough access rights to read the project owner."
        return User.get_by_id(self.owner_id)

    @classmethod
    def _from_api_project(cls, api_project: ApiProject) -> "Project":
        owner_id = None
        if api_project.owner is not None:
            owner_id = api_project.owner.id
        return cls(
            api_project.id,
            api_project.name,
            api_project.team,
            api_project.team_name,
            owner_id,
            api_project.priority,
            api_project.paused,
            api_project.expected_time,
        )
