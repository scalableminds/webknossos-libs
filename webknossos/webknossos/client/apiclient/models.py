from typing import List, Optional

import attr


@attr.s(auto_attribs=True)
class ApiShortLink:
    longLink: str


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
    displayName: Optional[str]
    description: Optional[str]
    isPublic: bool
    folderId: str
    allowedTeams: List[ApiTeam]
    tags: List[str]
    dataStore: ApiDataStore
