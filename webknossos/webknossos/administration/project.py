import warnings
from typing import TYPE_CHECKING

import attr

from ..client.api_client.models import ApiProject, ApiProjectCreate, ApiProjectUpdate
from ..client.context import _get_api_client
from .user import Team, User

if TYPE_CHECKING:
    from .task import Task


@attr.define
class Project:
    """Data class containing information about a WEBKNOSSOS project"""

    project_id: str
    name: str
    team_id: str
    team_name: str
    owner_id: str | None  # None in case you have no read access on the owner
    priority: int
    paused: bool
    expected_time: int | None
    is_blacklisted_from_report: bool

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

    @classmethod
    def create(
        cls,
        name: str,
        priority: int,
        paused: bool,
        expected_time: int | None,
        team: str | Team,
        is_blacklisted_from_report: bool,
        owner: str | User | None = None,
    ) -> "Project":
        """Creates a new project and returns it.
        Note: The project will be created with the current user as owner.

        Args:
            name (str): The name of the project.
            priority (int): The priority of the project.
            paused (bool): Whether the project is paused or not.
            expected_time (int | None): The expected time for the project in minutes.
            team_id (str): The ID of the team to which the project belongs.
            owner_id: (str | None): The ID of the owner of the project. If None, the current user will be used.

        Returns:
            Project: The created project.
        """
        if isinstance(team, Team):
            team = team.id
        if isinstance(owner, User):
            owner = owner.user_id
        api_client = _get_api_client(enforce_auth=True)
        api_project = ApiProjectCreate(
            name=name,
            team=team,
            priority=priority,
            paused=paused,
            expected_time=expected_time,
            owner=owner or User.get_current_user().user_id,
            is_blacklisted_from_report=is_blacklisted_from_report,
        )

        return cls._from_api_project(api_client.project_create(api_project))

    def delete(self) -> None:
        """Deletes this project. WARNING: This is irreversible!"""
        client = _get_api_client(enforce_auth=True)
        client.project_delete(self.project_id)

    def update(
        self,
        priority: int | None = None,
        expected_time: int | None = None,
        is_blacklisted_from_report: bool | None = None,
    ) -> "Project":
        """Updates the project with the given name and returns it."""
        api_project = ApiProjectUpdate(
            priority=priority if priority is not None else self.priority,
            expected_time=expected_time
            if expected_time is not None
            else self.expected_time,
            is_blacklisted_from_report=is_blacklisted_from_report
            if is_blacklisted_from_report is not None
            else self.is_blacklisted_from_report,
        )
        updated = _get_api_client(enforce_auth=True).project_update(
            self.project_id, api_project
        )

        return Project._from_api_project(updated)

    def get_tasks(self, fetch_all: bool = False) -> list["Task"]:
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
        assert self.owner_id is not None, (
            "Project owner is None, you may not have enough access rights to read the project owner."
        )
        return User.get_by_id(self.owner_id)

    @classmethod
    def _from_api_project(cls, api_project: ApiProject) -> "Project":
        owner_id = None
        if api_project.owner is not None:
            owner_id = api_project.owner.id
        return Project(
            project_id=api_project.id,
            name=api_project.name,
            team_id=api_project.team,
            team_name=api_project.team_name,
            owner_id=owner_id,
            priority=api_project.priority,
            paused=api_project.paused,
            expected_time=api_project.expected_time,
            is_blacklisted_from_report=api_project.is_blacklisted_from_report,
        )
