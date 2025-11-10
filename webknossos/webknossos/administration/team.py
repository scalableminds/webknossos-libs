from typing import TYPE_CHECKING

import attr

from ..client.api_client.models import ApiTeamAdd
from ..client.context import _get_api_client

if TYPE_CHECKING:
    from .user import User


@attr.frozen
class Team:
    id: str
    name: str
    organization_id: str

    @classmethod
    def get_by_name(cls, name: str) -> "Team":
        """Returns the Team specified by the passed name if your token authorizes you to see it."""
        client = _get_api_client()
        api_teams = client.team_list()
        for api_team in api_teams:
            if api_team.name == name:
                return cls(api_team.id, api_team.name, api_team.organization)
        raise KeyError(f"Could not find team {name}.")

    @classmethod
    def get_by_id(cls, team_id: str) -> "Team":
        """Returns the Team specified by the passed ID."""
        client = _get_api_client()
        api_teams = client.team_list()
        for api_team in api_teams:
            if api_team.id == team_id:
                return cls(api_team.id, api_team.name, api_team.organization)
        raise KeyError(f"Could not find team {team_id}.")

    @classmethod
    def get_list(cls) -> list["Team"]:
        """Returns all teams of the current user."""
        client = _get_api_client()
        api_teams = client.team_list()
        return [
            cls(api_team.id, api_team.name, api_team.organization)
            for api_team in api_teams
        ]

    @classmethod
    def add(cls, team_name: str) -> "Team":
        """Adds a new team with the specified name."""
        client = _get_api_client()
        client.team_add(team=ApiTeamAdd(team_name))
        return cls.get_by_name(team_name)

    def add_user(self, user: "User", *, is_team_manager: bool = False) -> None:
        """Adds a user to the team."""
        user.assign_team_roles(self, is_team_manager=is_team_manager)

    def delete(self) -> None:
        """Deletes the team."""
        client = _get_api_client()
        client.team_delete(team_id=self.id)
