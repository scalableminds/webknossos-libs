from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.annotation_infos_by_task_id_response_200_item_annotation_layers_item import (
        AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_data_store import (
        AnnotationInfosByTaskIdResponse200ItemDataStore,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_owner import (
        AnnotationInfosByTaskIdResponse200ItemOwner,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_restrictions import (
        AnnotationInfosByTaskIdResponse200ItemRestrictions,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_settings import (
        AnnotationInfosByTaskIdResponse200ItemSettings,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_stats import (
        AnnotationInfosByTaskIdResponse200ItemStats,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_task import (
        AnnotationInfosByTaskIdResponse200ItemTask,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_tracing_store import (
        AnnotationInfosByTaskIdResponse200ItemTracingStore,
    )
    from ..models.annotation_infos_by_task_id_response_200_item_user import (
        AnnotationInfosByTaskIdResponse200ItemUser,
    )


T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200Item")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200Item:
    """
    Attributes:
        state (str):
        id (str):
        name (str):
        description (str):
        typ (str):
        organization (str):
        data_store (AnnotationInfosByTaskIdResponse200ItemDataStore):
        tags (List[str]):
        modified (Union[Unset, int]):
        view_configuration (Union[Unset, str]):
        task (Union[Unset, None, AnnotationInfosByTaskIdResponse200ItemTask]):
        stats (Union[Unset, AnnotationInfosByTaskIdResponse200ItemStats]):
        restrictions (Union[Unset, AnnotationInfosByTaskIdResponse200ItemRestrictions]):
        formatted_hash (Union[Unset, str]):
        annotation_layers (Union[Unset, List['AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem']]):
        data_set_name (Union[Unset, str]):
        tracing_store (Union[Unset, AnnotationInfosByTaskIdResponse200ItemTracingStore]):
        visibility (Union[Unset, str]):
        settings (Union[Unset, AnnotationInfosByTaskIdResponse200ItemSettings]):
        tracing_time (Union[Unset, None, int]):
        teams (Union[Unset, List[Any]]):
        user (Union[Unset, AnnotationInfosByTaskIdResponse200ItemUser]):
        owner (Union[Unset, AnnotationInfosByTaskIdResponse200ItemOwner]):
        contributors (Union[Unset, List[Any]]):
        others_may_edit (Union[Unset, int]):
    """

    state: str
    id: str
    name: str
    description: str
    typ: str
    organization: str
    data_store: "AnnotationInfosByTaskIdResponse200ItemDataStore"
    tags: List[str]
    modified: Union[Unset, int] = UNSET
    view_configuration: Union[Unset, str] = UNSET
    task: Union[Unset, None, "AnnotationInfosByTaskIdResponse200ItemTask"] = UNSET
    stats: Union[Unset, "AnnotationInfosByTaskIdResponse200ItemStats"] = UNSET
    restrictions: Union[
        Unset, "AnnotationInfosByTaskIdResponse200ItemRestrictions"
    ] = UNSET
    formatted_hash: Union[Unset, str] = UNSET
    annotation_layers: Union[
        Unset, List["AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem"]
    ] = UNSET
    data_set_name: Union[Unset, str] = UNSET
    tracing_store: Union[
        Unset, "AnnotationInfosByTaskIdResponse200ItemTracingStore"
    ] = UNSET
    visibility: Union[Unset, str] = UNSET
    settings: Union[Unset, "AnnotationInfosByTaskIdResponse200ItemSettings"] = UNSET
    tracing_time: Union[Unset, None, int] = UNSET
    teams: Union[Unset, List[Any]] = UNSET
    user: Union[Unset, "AnnotationInfosByTaskIdResponse200ItemUser"] = UNSET
    owner: Union[Unset, "AnnotationInfosByTaskIdResponse200ItemOwner"] = UNSET
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
        from ..models.annotation_infos_by_task_id_response_200_item_annotation_layers_item import (
            AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_data_store import (
            AnnotationInfosByTaskIdResponse200ItemDataStore,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_owner import (
            AnnotationInfosByTaskIdResponse200ItemOwner,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_restrictions import (
            AnnotationInfosByTaskIdResponse200ItemRestrictions,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_settings import (
            AnnotationInfosByTaskIdResponse200ItemSettings,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_stats import (
            AnnotationInfosByTaskIdResponse200ItemStats,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_task import (
            AnnotationInfosByTaskIdResponse200ItemTask,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_tracing_store import (
            AnnotationInfosByTaskIdResponse200ItemTracingStore,
        )
        from ..models.annotation_infos_by_task_id_response_200_item_user import (
            AnnotationInfosByTaskIdResponse200ItemUser,
        )

        d = src_dict.copy()
        state = d.pop("state")

        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description")

        typ = d.pop("typ")

        organization = d.pop("organization")

        data_store = AnnotationInfosByTaskIdResponse200ItemDataStore.from_dict(
            d.pop("dataStore")
        )

        tags = cast(List[str], d.pop("tags"))

        modified = d.pop("modified", UNSET)

        view_configuration = d.pop("viewConfiguration", UNSET)

        _task = d.pop("task", UNSET)
        task: Union[Unset, None, AnnotationInfosByTaskIdResponse200ItemTask]
        if _task is None:
            task = None
        elif isinstance(_task, Unset):
            task = UNSET
        else:
            task = AnnotationInfosByTaskIdResponse200ItemTask.from_dict(_task)

        _stats = d.pop("stats", UNSET)
        stats: Union[Unset, AnnotationInfosByTaskIdResponse200ItemStats]
        if isinstance(_stats, Unset):
            stats = UNSET
        else:
            stats = AnnotationInfosByTaskIdResponse200ItemStats.from_dict(_stats)

        _restrictions = d.pop("restrictions", UNSET)
        restrictions: Union[Unset, AnnotationInfosByTaskIdResponse200ItemRestrictions]
        if isinstance(_restrictions, Unset):
            restrictions = UNSET
        else:
            restrictions = AnnotationInfosByTaskIdResponse200ItemRestrictions.from_dict(
                _restrictions
            )

        formatted_hash = d.pop("formattedHash", UNSET)

        annotation_layers = []
        _annotation_layers = d.pop("annotationLayers", UNSET)
        for annotation_layers_item_data in _annotation_layers or []:
            annotation_layers_item = (
                AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem.from_dict(
                    annotation_layers_item_data
                )
            )

            annotation_layers.append(annotation_layers_item)

        data_set_name = d.pop("dataSetName", UNSET)

        _tracing_store = d.pop("tracingStore", UNSET)
        tracing_store: Union[Unset, AnnotationInfosByTaskIdResponse200ItemTracingStore]
        if isinstance(_tracing_store, Unset):
            tracing_store = UNSET
        else:
            tracing_store = (
                AnnotationInfosByTaskIdResponse200ItemTracingStore.from_dict(
                    _tracing_store
                )
            )

        visibility = d.pop("visibility", UNSET)

        _settings = d.pop("settings", UNSET)
        settings: Union[Unset, AnnotationInfosByTaskIdResponse200ItemSettings]
        if isinstance(_settings, Unset):
            settings = UNSET
        else:
            settings = AnnotationInfosByTaskIdResponse200ItemSettings.from_dict(
                _settings
            )

        tracing_time = d.pop("tracingTime", UNSET)

        teams = cast(List[Any], d.pop("teams", UNSET))

        _user = d.pop("user", UNSET)
        user: Union[Unset, AnnotationInfosByTaskIdResponse200ItemUser]
        if isinstance(_user, Unset):
            user = UNSET
        else:
            user = AnnotationInfosByTaskIdResponse200ItemUser.from_dict(_user)

        _owner = d.pop("owner", UNSET)
        owner: Union[Unset, AnnotationInfosByTaskIdResponse200ItemOwner]
        if isinstance(_owner, Unset):
            owner = UNSET
        else:
            owner = AnnotationInfosByTaskIdResponse200ItemOwner.from_dict(_owner)

        contributors = cast(List[Any], d.pop("contributors", UNSET))

        others_may_edit = d.pop("othersMayEdit", UNSET)

        annotation_infos_by_task_id_response_200_item = cls(
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

        annotation_infos_by_task_id_response_200_item.additional_properties = d
        return annotation_infos_by_task_id_response_200_item

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
