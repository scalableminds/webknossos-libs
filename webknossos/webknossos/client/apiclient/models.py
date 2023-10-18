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
class ApiUploadInformation:
    upload_id: str
