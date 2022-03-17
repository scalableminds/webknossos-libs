from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.annotation_info_response_200_annotation_layers_item import (
    AnnotationInfoResponse200AnnotationLayersItem,
)
from ..models.annotation_info_response_200_data_store import (
    AnnotationInfoResponse200DataStore,
)
from ..models.annotation_info_response_200_restrictions import (
    AnnotationInfoResponse200Restrictions,
)
from ..models.annotation_info_response_200_settings import (
    AnnotationInfoResponse200Settings,
)
from ..models.annotation_info_response_200_stats import AnnotationInfoResponse200Stats
from ..models.annotation_info_response_200_task import AnnotationInfoResponse200Task
from ..models.annotation_info_response_200_tracing_store import (
    AnnotationInfoResponse200TracingStore,
)
from ..models.annotation_info_response_200_user import AnnotationInfoResponse200User
from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfoResponse200")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200:
    """ """

    modified: int
    state: str
    id: str
    name: str
    description: str
    typ: str
    stats: AnnotationInfoResponse200Stats
    restrictions: AnnotationInfoResponse200Restrictions
    formatted_hash: str
    annotation_layers: List[AnnotationInfoResponse200AnnotationLayersItem]
    data_set_name: str
    organization: str
    data_store: AnnotationInfoResponse200DataStore
    tracing_store: AnnotationInfoResponse200TracingStore
    visibility: str
    settings: AnnotationInfoResponse200Settings
    tags: List[str]
    user: AnnotationInfoResponse200User
    meshes: List[Any]
    task: Optional[AnnotationInfoResponse200Task]
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

        stats = AnnotationInfoResponse200Stats.from_dict(d.pop("stats"))

        restrictions = AnnotationInfoResponse200Restrictions.from_dict(
            d.pop("restrictions")
        )

        formatted_hash = d.pop("formattedHash")

        annotation_layers = []
        _annotation_layers = d.pop("annotationLayers")
        for annotation_layers_item_data in _annotation_layers:
            annotation_layers_item = (
                AnnotationInfoResponse200AnnotationLayersItem.from_dict(
                    annotation_layers_item_data
                )
            )

            annotation_layers.append(annotation_layers_item)

        data_set_name = d.pop("dataSetName")

        organization = d.pop("organization")

        data_store = AnnotationInfoResponse200DataStore.from_dict(d.pop("dataStore"))

        tracing_store = AnnotationInfoResponse200TracingStore.from_dict(
            d.pop("tracingStore")
        )

        visibility = d.pop("visibility")

        settings = AnnotationInfoResponse200Settings.from_dict(d.pop("settings"))

        tags = cast(List[str], d.pop("tags"))

        user = AnnotationInfoResponse200User.from_dict(d.pop("user"))

        meshes = []
        _meshes = d.pop("meshes")
        for meshes_item_data in _meshes:
            meshes_item = meshes_item_data

            meshes.append(meshes_item)

        view_configuration = d.pop("viewConfiguration", UNSET)

        _task = d.pop("task")
        task: Optional[AnnotationInfoResponse200Task]
        if _task is None:
            task = None
        else:
            task = AnnotationInfoResponse200Task.from_dict(_task)

        tracing_time = d.pop("tracingTime")

        annotation_info_response_200 = cls(
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
