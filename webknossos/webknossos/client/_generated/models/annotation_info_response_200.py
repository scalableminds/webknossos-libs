from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.annotation_info_response_200_annotation_layers_item import (
        AnnotationInfoResponse200AnnotationLayersItem,
    )
    from ..models.annotation_info_response_200_data_store import (
        AnnotationInfoResponse200DataStore,
    )
    from ..models.annotation_info_response_200_owner import (
        AnnotationInfoResponse200Owner,
    )
    from ..models.annotation_info_response_200_restrictions import (
        AnnotationInfoResponse200Restrictions,
    )
    from ..models.annotation_info_response_200_settings import (
        AnnotationInfoResponse200Settings,
    )
    from ..models.annotation_info_response_200_stats import (
        AnnotationInfoResponse200Stats,
    )
    from ..models.annotation_info_response_200_task import AnnotationInfoResponse200Task
    from ..models.annotation_info_response_200_tracing_store import (
        AnnotationInfoResponse200TracingStore,
    )
    from ..models.annotation_info_response_200_user import AnnotationInfoResponse200User


T = TypeVar("T", bound="AnnotationInfoResponse200")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200:
    """
    Attributes:
        state (str):
        id (str):
        name (str):
        description (str):
        typ (str):
        organization (str):
        data_store (AnnotationInfoResponse200DataStore):
        tags (List[str]):
        modified (Union[Unset, int]):
        view_configuration (Union[Unset, str]):
        task (Union[Unset, None, AnnotationInfoResponse200Task]):
        stats (Union[Unset, AnnotationInfoResponse200Stats]):
        restrictions (Union[Unset, AnnotationInfoResponse200Restrictions]):
        formatted_hash (Union[Unset, str]):
        annotation_layers (Union[Unset, List['AnnotationInfoResponse200AnnotationLayersItem']]):
        data_set_name (Union[Unset, str]):
        tracing_store (Union[Unset, AnnotationInfoResponse200TracingStore]):
        visibility (Union[Unset, str]):
        settings (Union[Unset, AnnotationInfoResponse200Settings]):
        tracing_time (Union[Unset, None, int]):
        teams (Union[Unset, List[Any]]):
        user (Union[Unset, AnnotationInfoResponse200User]):
        owner (Union[Unset, AnnotationInfoResponse200Owner]):
        contributors (Union[Unset, List[Any]]):
        others_may_edit (Union[Unset, int]):
    """

    state: str
    id: str
    name: str
    description: str
    typ: str
    organization: str
    data_store: "AnnotationInfoResponse200DataStore"
    tags: List[str]
    modified: Union[Unset, int] = UNSET
    view_configuration: Union[Unset, str] = UNSET
    task: Union[Unset, None, "AnnotationInfoResponse200Task"] = UNSET
    stats: Union[Unset, "AnnotationInfoResponse200Stats"] = UNSET
    restrictions: Union[Unset, "AnnotationInfoResponse200Restrictions"] = UNSET
    formatted_hash: Union[Unset, str] = UNSET
    annotation_layers: Union[
        Unset, List["AnnotationInfoResponse200AnnotationLayersItem"]
    ] = UNSET
    data_set_name: Union[Unset, str] = UNSET
    tracing_store: Union[Unset, "AnnotationInfoResponse200TracingStore"] = UNSET
    visibility: Union[Unset, str] = UNSET
    settings: Union[Unset, "AnnotationInfoResponse200Settings"] = UNSET
    tracing_time: Union[Unset, None, int] = UNSET
    teams: Union[Unset, List[Any]] = UNSET
    user: Union[Unset, "AnnotationInfoResponse200User"] = UNSET
    owner: Union[Unset, "AnnotationInfoResponse200Owner"] = UNSET
    contributors: Union[Unset, List[Any]] = UNSET
    others_may_edit: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        state = self.state
        id = self.id
        name = self.name
        description = self.description
        typ = self.typ
        organization = self.organization
        data_store = self.data_store.to_dict()

        tags = self.tags

        modified = self.modified
        view_configuration = self.view_configuration
        task: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.task, Unset):
            task = self.task.to_dict() if self.task else None

        stats: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.stats, Unset):
            stats = self.stats.to_dict()

        restrictions: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.restrictions, Unset):
            restrictions = self.restrictions.to_dict()

        formatted_hash = self.formatted_hash
        annotation_layers: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.annotation_layers, Unset):
            annotation_layers = []
            for annotation_layers_item_data in self.annotation_layers:
                annotation_layers_item = annotation_layers_item_data.to_dict()

                annotation_layers.append(annotation_layers_item)

        data_set_name = self.data_set_name
        tracing_store: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.tracing_store, Unset):
            tracing_store = self.tracing_store.to_dict()

        visibility = self.visibility
        settings: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.settings, Unset):
            settings = self.settings.to_dict()

        tracing_time = self.tracing_time
        teams: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.teams, Unset):
            teams = self.teams

        user: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.user, Unset):
            user = self.user.to_dict()

        owner: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.owner, Unset):
            owner = self.owner.to_dict()

        contributors: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.contributors, Unset):
            contributors = self.contributors

        others_may_edit = self.others_may_edit

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "state": state,
                "id": id,
                "name": name,
                "description": description,
                "typ": typ,
                "organization": organization,
                "dataStore": data_store,
                "tags": tags,
            }
        )
        if modified is not UNSET:
            field_dict["modified"] = modified
        if view_configuration is not UNSET:
            field_dict["viewConfiguration"] = view_configuration
        if task is not UNSET:
            field_dict["task"] = task
        if stats is not UNSET:
            field_dict["stats"] = stats
        if restrictions is not UNSET:
            field_dict["restrictions"] = restrictions
        if formatted_hash is not UNSET:
            field_dict["formattedHash"] = formatted_hash
        if annotation_layers is not UNSET:
            field_dict["annotationLayers"] = annotation_layers
        if data_set_name is not UNSET:
            field_dict["dataSetName"] = data_set_name
        if tracing_store is not UNSET:
            field_dict["tracingStore"] = tracing_store
        if visibility is not UNSET:
            field_dict["visibility"] = visibility
        if settings is not UNSET:
            field_dict["settings"] = settings
        if tracing_time is not UNSET:
            field_dict["tracingTime"] = tracing_time
        if teams is not UNSET:
            field_dict["teams"] = teams
        if user is not UNSET:
            field_dict["user"] = user
        if owner is not UNSET:
            field_dict["owner"] = owner
        if contributors is not UNSET:
            field_dict["contributors"] = contributors
        if others_may_edit is not UNSET:
            field_dict["othersMayEdit"] = others_may_edit

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.annotation_info_response_200_annotation_layers_item import (
            AnnotationInfoResponse200AnnotationLayersItem,
        )
        from ..models.annotation_info_response_200_data_store import (
            AnnotationInfoResponse200DataStore,
        )
        from ..models.annotation_info_response_200_owner import (
            AnnotationInfoResponse200Owner,
        )
        from ..models.annotation_info_response_200_restrictions import (
            AnnotationInfoResponse200Restrictions,
        )
        from ..models.annotation_info_response_200_settings import (
            AnnotationInfoResponse200Settings,
        )
        from ..models.annotation_info_response_200_stats import (
            AnnotationInfoResponse200Stats,
        )
        from ..models.annotation_info_response_200_task import (
            AnnotationInfoResponse200Task,
        )
        from ..models.annotation_info_response_200_tracing_store import (
            AnnotationInfoResponse200TracingStore,
        )
        from ..models.annotation_info_response_200_user import (
            AnnotationInfoResponse200User,
        )

        d = src_dict.copy()
        state = d.pop("state")

        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description")

        typ = d.pop("typ")

        organization = d.pop("organization")

        data_store = AnnotationInfoResponse200DataStore.from_dict(d.pop("dataStore"))

        tags = cast(List[str], d.pop("tags"))

        modified = d.pop("modified", UNSET)

        view_configuration = d.pop("viewConfiguration", UNSET)

        _task = d.pop("task", UNSET)
        task: Union[Unset, None, AnnotationInfoResponse200Task]
        if _task is None:
            task = None
        elif isinstance(_task, Unset):
            task = UNSET
        else:
            task = AnnotationInfoResponse200Task.from_dict(_task)

        _stats = d.pop("stats", UNSET)
        stats: Union[Unset, AnnotationInfoResponse200Stats]
        if isinstance(_stats, Unset):
            stats = UNSET
        else:
            stats = AnnotationInfoResponse200Stats.from_dict(_stats)

        _restrictions = d.pop("restrictions", UNSET)
        restrictions: Union[Unset, AnnotationInfoResponse200Restrictions]
        if isinstance(_restrictions, Unset):
            restrictions = UNSET
        else:
            restrictions = AnnotationInfoResponse200Restrictions.from_dict(
                _restrictions
            )

        formatted_hash = d.pop("formattedHash", UNSET)

        annotation_layers = []
        _annotation_layers = d.pop("annotationLayers", UNSET)
        for annotation_layers_item_data in _annotation_layers or []:
            annotation_layers_item = (
                AnnotationInfoResponse200AnnotationLayersItem.from_dict(
                    annotation_layers_item_data
                )
            )

            annotation_layers.append(annotation_layers_item)

        data_set_name = d.pop("dataSetName", UNSET)

        _tracing_store = d.pop("tracingStore", UNSET)
        tracing_store: Union[Unset, AnnotationInfoResponse200TracingStore]
        if isinstance(_tracing_store, Unset):
            tracing_store = UNSET
        else:
            tracing_store = AnnotationInfoResponse200TracingStore.from_dict(
                _tracing_store
            )

        visibility = d.pop("visibility", UNSET)

        _settings = d.pop("settings", UNSET)
        settings: Union[Unset, AnnotationInfoResponse200Settings]
        if isinstance(_settings, Unset):
            settings = UNSET
        else:
            settings = AnnotationInfoResponse200Settings.from_dict(_settings)

        tracing_time = d.pop("tracingTime", UNSET)

        teams = cast(List[Any], d.pop("teams", UNSET))

        _user = d.pop("user", UNSET)
        user: Union[Unset, AnnotationInfoResponse200User]
        if isinstance(_user, Unset):
            user = UNSET
        else:
            user = AnnotationInfoResponse200User.from_dict(_user)

        _owner = d.pop("owner", UNSET)
        owner: Union[Unset, AnnotationInfoResponse200Owner]
        if isinstance(_owner, Unset):
            owner = UNSET
        else:
            owner = AnnotationInfoResponse200Owner.from_dict(_owner)

        contributors = cast(List[Any], d.pop("contributors", UNSET))

        others_may_edit = d.pop("othersMayEdit", UNSET)

        annotation_info_response_200 = cls(
            state=state,
            id=id,
            name=name,
            description=description,
            typ=typ,
            organization=organization,
            data_store=data_store,
            tags=tags,
            modified=modified,
            view_configuration=view_configuration,
            task=task,
            stats=stats,
            restrictions=restrictions,
            formatted_hash=formatted_hash,
            annotation_layers=annotation_layers,
            data_set_name=data_set_name,
            tracing_store=tracing_store,
            visibility=visibility,
            settings=settings,
            tracing_time=tracing_time,
            teams=teams,
            user=user,
            owner=owner,
            contributors=contributors,
            others_may_edit=others_may_edit,
        )

        annotation_info_response_200.additional_properties = d
        return annotation_info_response_200

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
