from typing import Dict, List, Tuple

import attr

from ..client.api_client.models import ApiLoggedTimeGroupedByMonth, ApiUser
from ..client.context import _get_api_client


@attr.frozen
class User:
    """Represents a user of a webknossos instance.
    You can get users via `get_current_user` and `get_all_managed_users`."""

    user_id: str
    email: str
    organization_id: str
    first_name: str
    last_name: str
    created: int
    last_activity: int
    teams: Tuple["Team", ...]
    experiences: Dict[str, int]
    is_active: bool
    is_admin: bool
    is_dataset_manager: bool

    def get_logged_times(self) -> List["LoggedTime"]:
        """Get the logged times of this user.
        Returns a list of `LoggedTime` objects where one represents one month."""
        client = _get_api_client(enforce_auth=True)
        api_logged_times: ApiLoggedTimeGroupedByMonth = client.user_logged_time(
            self.user_id
        )
        return [
            LoggedTime(
                duration_in_seconds=i.duration_in_seconds,
                month=i.payment_interval.month,
                year=i.payment_interval.year,
            )
            for i in api_logged_times.logged_time
        ]

    @classmethod
    def _from_api_user(cls, api_user: ApiUser) -> "User":
        return cls(
            user_id=api_user.id,
            email=api_user.email,
            organization_id=api_user.organization,
            first_name=api_user.first_name,
            last_name=api_user.last_name,
            created=api_user.created,
            last_activity=api_user.last_activity,
            teams=tuple(
                Team(id=team.id, name=team.name, organization_id=api_user.organization)
                for team in api_user.teams
            ),
            experiences=api_user.experiences,
            is_active=api_user.is_active,
            is_admin=api_user.is_admin,
            is_dataset_manager=api_user.is_dataset_manager,
        )

    @classmethod
    def get_by_id(cls, id: str) -> "User":  # pylint: disable=redefined-builtin
        """Returns the user specified by the passed id if your token authorizes you to see them."""
        client = _get_api_client(enforce_auth=True)
        api_user = client.user_by_id(id)
        print(api_user)
        return cls._from_api_user(api_user)

    @classmethod
    def get_current_user(cls) -> "User":
        """Returns the current user from the authentication context."""
        client = _get_api_client(enforce_auth=True)
        api_user = client.user_current()
        return cls._from_api_user(api_user)

    @classmethod
    def get_all_managed_users(cls) -> List["User"]:
        """Returns all users of whom the current user is admin or team-manager."""
        client = _get_api_client(enforce_auth=True)
        api_users = client.user_list()
        return [cls._from_api_user(i) for i in api_users]


@attr.frozen
class Team:
    id: str
    name: str
    organization_id: str

    @classmethod
    def get_by_name(cls, name: str) -> "Team":
        """Returns the Team specified by the passed name if your token authorizes you to see it."""
        client = _get_api_client(enforce_auth=True)
        api_teams = client.team_list()
        for api_team in api_teams:
            if api_team.name == name:
                return cls(api_team.id, api_team.name, api_team.organization)
        raise KeyError(f"Could not find team {name}.")


@attr.frozen
class LoggedTime:
    duration_in_seconds: int
    year: int
    month: int
