import attr
from typing import List

@attr.s(auto_attribs=True)
class ApiShortLink:
    longLink: str


@attr.s(auto_attribs=True)
class ApiDataStore:
    url: str


@attr.s(auto_attribs=True)
class ApiDataset:
    name: str
    tags: List[str]
    dataStore: ApiDataStore
