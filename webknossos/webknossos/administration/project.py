import warnings
from typing import TYPE_CHECKING

import attr

from ..client.api_client.errors import UnexpectedStatusError
from ..client.api_client.models import ApiProject, ApiProjectCreate
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
        """
        Retrieve a project by its unique identifier.

        Fetches the project with the specified `project_id` from the backend, provided the current
        user can access the project. This method requires valid authentication.

        Args:
            project_id (str): The unique identifier of the project to retrieve.

        Returns:
            Project: An instance of the Project class corresponding to the specified ID.

        Raises:
            UnexpectedStatusError: If the project does not exist or the user is not authorized to access it.

        Examples:
            Retrieve a project by ID:
                ```
                project = Project.get_by_id("project_12345")
                print(project.name)
                ```
        """
        api_project = _get_api_client(enforce_auth=True).project_info_by_id(project_id)
        return cls._from_api_project(api_project)

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        """
        Retrieve a project by its unique name.

        Fetches the project with the specified `name` from the backend, provided the current
        user can access the project. This method requires valid authentication.

        Args:
            name (str): The unique name of the project to retrieve.

        Returns:
            Project: An instance of the Project class corresponding to the specified name.

        Raises:
            UnexpectedStatusError: If the project does not exist or the user is not authorized to access it.

        Examples:
            Retrieve a project by name:
                ```
                project = Project.get_by_name("my_project")
                print(project.project_id)
                ```
        """
        api_client = _get_api_client(enforce_auth=True)
        try:
            api_project = api_client.project_info_by_name(name)
        except UnexpectedStatusError as e:
            if "Project could not be found" in str(e):
                raise ValueError(f"Project with name '{name}' does not exist.")
            raise e
        return cls._from_api_project(api_project)

    @classmethod
    def create(
        cls,
        name: str,
        team: str | Team,
        expected_time: int | None,
        priority: int = 100,
        paused: bool = False,
        is_blacklisted_from_report: bool = False,
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
            owner_id: (str | None): The ID of the owner user of the project. If None, the current user will be used.

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
        """Deletes this project from server. WARNING: This is irreversible!"""
        client = _get_api_client(enforce_auth=True)
        client.project_delete(self.project_id)

    def update(
        self,
        priority: int | None = None,
        expected_time: int | None = None,
        is_blacklisted_from_report: bool | None = None,
    ) -> "Project":
        """
        Update the project's properties and return the updated Project instance.

        This method updates the project's priority, expected time, and blacklist status for reporting.
        Only the parameters provided (not None) will be updated; others remain unchanged.

        Args:
            priority (int | None): Optional new priority for the project. If not provided, the current priority is kept.
            expected_time (int | None): Optional new expected time (in minutes or relevant unit) for the project. If not provided, the current expected time is kept.
            is_blacklisted_from_report (bool | None): Optional flag to set whether the project should be excluded from reports. If not provided, the current blacklist status is kept.

        Returns:
            Project: The updated Project instance reflecting the new properties.

        Examples:
            Update only the priority:
                ```
                project = project.update(priority=5)
                ```

            Update expected time and blacklist status:
                ```
                project = project.update(expected_time=120, is_blacklisted_from_report=True)
                ```
        """
        api_project = ApiProjectCreate(
            name=self.name,
            team=self.team_id,
            priority=priority if priority is not None else self.priority,
            paused=self.paused,
            expected_time=expected_time
            if expected_time is not None
            else self.expected_time,
            is_blacklisted_from_report=is_blacklisted_from_report
            if is_blacklisted_from_report is not None
            else self.is_blacklisted_from_report,
            owner=self.owner_id,
        )
        updated = _get_api_client(enforce_auth=True).project_update(
            self.project_id, api_project
        )
        return Project._from_api_project(updated)

    def get_tasks(self, fetch_all: bool = False) -> list["Task"]:
        """Retrieve the list of tasks associated with this project.

        By default, this method fetches up to the first 1000 tasks for the project. If the project contains more than 1000 tasks,
        a warning is issued unless `fetch_all=True` is passed, in which case all tasks are fetched using pagination.

        Args:
            fetch_all (bool, optional): If True, fetches all tasks for the project using pagination. If False (default),
                only the first 1000 tasks are returned and a warning is issued if more tasks exist.

        Returns:
            list[Task]: A list of Task objects associated with this project.

        Raises:
            Any exceptions raised by the underlying API client.

        Examples:
            Fetch up to 1000 tasks:
                ```
                tasks = project.get_tasks()
                print(f"Fetched {len(tasks)} tasks")
                ```

            Fetch all tasks, regardless of count:
                ```
                all_tasks = project.get_tasks(fetch_all=True)
                print(f"Total tasks: {len(all_tasks)}")
                ```
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
        return cls(
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
