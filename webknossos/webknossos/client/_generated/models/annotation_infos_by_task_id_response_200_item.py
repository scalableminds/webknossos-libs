from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.annotation_infos_by_task_id_response_200_item_annotation_layers_item import (
    AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem,
)
from ..models.annotation_infos_by_task_id_response_200_item_data_store import (
    AnnotationInfosByTaskIdResponse200ItemDataStore,
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
from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200Item")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200Item:
    """ """

    modified: int
    state: str
    id: str
    name: str
    description: str
    typ: str
    stats: AnnotationInfosByTaskIdResponse200ItemStats
    restrictions: AnnotationInfosByTaskIdResponse200ItemRestrictions
    formatted_hash: str
    annotation_layers: List[AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem]
    data_set_name: str
    organization: str
    data_store: AnnotationInfosByTaskIdResponse200ItemDataStore
    tracing_store: AnnotationInfosByTaskIdResponse200ItemTracingStore
    visibility: str
    settings: AnnotationInfosByTaskIdResponse200ItemSettings
    tags: List[str]
    user: AnnotationInfosByTaskIdResponse200ItemUser
    meshes: List[Any]
    task: Optional[AnnotationInfosByTaskIdResponse200ItemTask]
    tracing_time: Optional[int]
    view_configuration: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        modified = self.modified
        state = self.state
        id = self.id
        name = self.name
        description = self.description
        typ = self.typ
        stats = self.stats.to_dict()

        restrictions = self.restrictions.to_dict()

        formatted_hash = self.formatted_hash
        annotation_layers = []
        for annotation_layers_item_data in self.annotation_layers:
            annotation_layers_item = annotation_layers_item_data.to_dict()

            annotation_layers.append(annotation_layers_item)

        data_set_name = self.data_set_name
        organization = self.organization
        data_store = self.data_store.to_dict()

        tracing_store = self.tracing_store.to_dict()

        visibility = self.visibility
        settings = self.settings.to_dict()

        tags = self.tags

        user = self.user.to_dict()

        meshes = []
        for meshes_item_data in self.meshes:
            meshes_item = meshes_item_data

            meshes.append(meshes_item)

        view_configuration = self.view_configuration
        task = self.task.to_dict() if self.task else None

        tracing_time = self.tracing_time

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "modified": modified,
                "state": state,
                "id": id,
                "name": name,
                "description": description,
                "typ": typ,
                "stats": stats,
                "restrictions": restrictions,
                "formattedHash": formatted_hash,
                "annotationLayers": annotation_layers,
                "dataSetName": data_set_name,
                "organization": organization,
                "dataStore": data_store,
                "tracingStore": tracing_store,
                "visibility": visibility,
                "settings": settings,
                "tags": tags,
                "user": user,
                "meshes": meshes,
                "task": task,
                "tracingTime": tracing_time,
            }
        )
        if view_configuration is not UNSET:
            field_dict["viewConfiguration"] = view_configuration

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        modified = d.pop("modified")

        state = d.pop("state")

        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description")

        typ = d.pop("typ")

        stats = AnnotationInfosByTaskIdResponse200ItemStats.from_dict(d.pop("stats"))

        restrictions = AnnotationInfosByTaskIdResponse200ItemRestrictions.from_dict(
            d.pop("restrictions")
        )

        formatted_hash = d.pop("formattedHash")

        annotation_layers = []
        _annotation_layers = d.pop("annotationLayers")
        for annotation_layers_item_data in _annotation_layers:
            annotation_layers_item = (
                AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem.from_dict(
                    annotation_layers_item_data
                )
            )

            annotation_layers.append(annotation_layers_item)

        data_set_name = d.pop("dataSetName")

        organization = d.pop("organization")

        data_store = AnnotationInfosByTaskIdResponse200ItemDataStore.from_dict(
            d.pop("dataStore")
        )

        tracing_store = AnnotationInfosByTaskIdResponse200ItemTracingStore.from_dict(
            d.pop("tracingStore")
        )

        visibility = d.pop("visibility")

        settings = AnnotationInfosByTaskIdResponse200ItemSettings.from_dict(
            d.pop("settings")
        )

        tags = cast(List[str], d.pop("tags"))

        user = AnnotationInfosByTaskIdResponse200ItemUser.from_dict(d.pop("user"))

        meshes = []
        _meshes = d.pop("meshes")
        for meshes_item_data in _meshes:
            meshes_item = meshes_item_data

            meshes.append(meshes_item)

        view_configuration = d.pop("viewConfiguration", UNSET)

        _task = d.pop("task")
        task: Optional[AnnotationInfosByTaskIdResponse200ItemTask]
        if _task is None:
            task = None
        else:
            task = AnnotationInfosByTaskIdResponse200ItemTask.from_dict(_task)

        tracing_time = d.pop("tracingTime")

        annotation_infos_by_task_id_response_200_item = cls(
            modified=modified,
            state=state,
            id=id,
            name=name,
            description=description,
            typ=typ,
            stats=stats,
            restrictions=restrictions,
            formatted_hash=formatted_hash,
            annotation_layers=annotation_layers,
            data_set_name=data_set_name,
            organization=organization,
            data_store=data_store,
            tracing_store=tracing_store,
            visibility=visibility,
            settings=settings,
            tags=tags,
            user=user,
            meshes=meshes,
            view_configuration=view_configuration,
            task=task,
            tracing_time=tracing_time,
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
