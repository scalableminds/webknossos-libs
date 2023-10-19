from typing import List, Optional

import attr


@attr.s(auto_attribs=True)
class ApiShortLink:
    long_link: str


@attr.s(auto_attribs=True)
class ApiDataStore:
    url: str


@attr.s(auto_attribs=True)
class ApiTeam:
    id: str
    name: str
    organization: str


@attr.s(auto_attribs=True)
class ApiDataset:
    name: str
    display_name: Optional[str]
    description: Optional[str]
    is_public: bool
    folder_id: str
    allowed_teams: List[ApiTeam]
    tags: List[str]
    data_store: ApiDataStore


@attr.s(auto_attribs=True)
class ApiSharingToken:
    sharing_token: str


@attr.s(auto_attribs=True)
class ApiDatastore:
    name: str
    allows_upload: bool
    url: str


@attr.s(auto_attribs=True)
class ApiUploadInformation:
    upload_id: str


@attr.s(auto_attribs=True)
class ApiLinkedLayerIdentifier:
    organization_name: str
    data_set_name: str
    layer_name: str
    new_layer_name: Optional[str]


@attr.s(auto_attribs=True)
class ApiReserveUploadInformation:
    upload_id: str
    name: str
    organization: str
    total_file_count: int
    layers_to_link: Optional[List[ApiLinkedLayerIdentifier]]
    initial_teams: List[str]
    folder_id: Optional[str]


@attr.s(auto_attribs=True)
class ApiTask:
    id: str


@attr.s(auto_attribs=True)
class ApiUser:
    id: str


@attr.s(auto_attribs=True)
class ApiProject:
    id: str
    name: str
    team: str
    team_name: str
    owner: ApiUser
    priority: int
    paused: bool
    expected_time: Optional[int]