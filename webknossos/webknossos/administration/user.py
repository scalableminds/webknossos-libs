from typing import TYPE_CHECKING, Dict, List, Union

import attr

from webknossos.client._generated.api.default import (
    current_user_info,
    user_info_by_id,
    user_list,
    user_logged_time,
)

if TYPE_CHECKING:
    from webknossos.client._generated.models.current_user_info_response_200 import (
        CurrentUserInfoResponse200,
    )
    from webknossos.client._generated.models.user_list_response_200_item import (
        UserListResponse200Item,
    )
    from webknossos.client._generated.models.user_info_by_id_response_200 import (
        UserInfoByIdResponse200,
    )

from webknossos.client.context import _get_generated_client


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
    teams: List["Team"]
    experiences: Dict[str, int]
    is_active: bool
    is_admin: bool
    is_dataset_manager: bool

    def get_logged_times(self) -> List["LoggedTime"]:
        """Get the logged times of this user.
        Returns a list of `LoggedTime` objects where one represents one month."""
        client = _get_generated_client(enforce_auth=True)
        response = user_logged_time.sync(id=self.user_id, client=client)
        assert response is not None, f"Could not fetch logged time of {self}"
        return [
            LoggedTime(
                duration_in_seconds=i.duration_in_seconds,
                month=i.payment_interval.month,
                year=i.payment_interval.year,
            )
            for i in response.logged_time
        ]

    @classmethod
    def _from_generated_response(
        cls,
        response: Union[
            "UserListResponse200Item",
            "CurrentUserInfoResponse200",
            "UserInfoByIdResponse200",
        ],
    ) -> "User":
        return cls(
            user_id=response.id,
            email=response.email,
            organization_id=response.organization,
            first_name=response.first_name,
            last_name=response.last_name,
            created=response.created,
            last_activity=response.last_activity,
            teams=[Team(id=team.id, name=team.name) for team in response.teams],
            experiences=response.experiences.additional_properties,
            is_active=bool(response.is_active),
            is_admin=bool(response.is_admin),
            is_dataset_manager=bool(response.is_dataset_manager),
        )

    @classmethod
    def get_by_id(cls, id: str) -> "User":  # pylint: disable=redefined-builtin
        """Returns the user specified by the passed id if your token authorizes you to see them."""
        client = _get_generated_client(enforce_auth=True)
        response = user_info_by_id.sync(id, client=client)
        assert response is not None, "Could not fetch user by id."
        return cls._from_generated_response(response)

    @classmethod
    def get_current_user(cls) -> "User":
        """Returns the current user from the authentication context."""
        client = _get_generated_client(enforce_auth=True)
        response = current_user_info.sync(client=client)
        assert response is not None, "Could not fetch current user."
        return cls._from_generated_response(response)

    @classmethod
    def get_all_managed_users(cls) -> List["User"]:
        """Returns all users of whom the current user is admin or team-manager."""
        client = _get_generated_client(enforce_auth=True)
        response = user_list.sync(client=client)
        assert response is not None, "Could not fetch managed users."
        return [cls._from_generated_response(i) for i in response]


@attr.frozen
class Team:
    id: str
    name: str


@attr.frozen
class LoggedTime:
    duration_in_seconds: int
    year: int
    month: int
