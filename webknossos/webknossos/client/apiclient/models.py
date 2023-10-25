from typing import List, Optional, Tuple

import attr


@attr.s(auto_attribs=True)
class ApiWkBuildInfoWebknossos:
    name: str
    version: str


@attr.s(auto_attribs=True)
class ApiWkBuildInfo:
    webknossos: ApiWkBuildInfoWebknossos
    local_data_store_enabled: bool
    local_tracing_store_enabled: bool


@attr.s(auto_attribs=True)
class ApiShortLink:
    long_link: str


@attr.s(auto_attribs=True)
class ApiDataStore:
    url: str
    name: str
    allows_upload: bool


@attr.s(auto_attribs=True)
class ApiTeam:
    id: str
    name: str
    organization: str


@attr.s(auto_attribs=True)
class ApiBoundingBox:
    top_left: Tuple[int, int, int]
    width: int
    height: int
    depth: int


@attr.s(auto_attribs=True)
class ApiDataLayer:
    name: str
    category: str
    element_class: str
    largest_segment_id: Optional[int]
    bounding_box: ApiBoundingBox


@attr.s(auto_attribs=True)
class ApiDataSource:
    data_layers: Optional[List[ApiDataLayer]]
    status: Optional[str]
    scale: Optional[Tuple[float, float, float]]


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
    data_source: ApiDataSource


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
class ApiTaskStatus:
    pending: int
    active: int
    finished: int


@attr.s(auto_attribs=True)
class ApiTaskType:
    id: str
    summary: str
    description: str
    team_id: str
    team_name: str


@attr.s(auto_attribs=True)
class ApiTask:
    id: str
    project_id: str
    data_set: str
    status: ApiTaskStatus
    type: ApiTaskType


@attr.s(auto_attribs=True)
class ApiTaskParameters:
    taks_type_id: str
    needed_experience: ApiExperience
    pending_nstances: int
    project_name: str
    script_id: str
    bounding_box: ApiBoundingBox


@attr.s(auto_attribs=True)
class ApiTeamMembership:
    id: str
    name: str


@attr.s(auto_attribs=True)
class ApiExperience:
    domain: str
    value: int
    # TODO adapt API to this?


@attr.s(auto_attribs=True)
class ApiUser:
    id: str
    email: str
    organization: str
    first_name: str
    last_name: str
    created: int
    last_activity: int
    is_active: bool
    is_admin: bool
    is_dataset_manager: bool
    teams: List[ApiTeamMembership]


@attr.s(auto_attribs=True)
class ApiMonth:
    month: int
    year: int


@attr.s(auto_attribs=True)
class ApiLoggedTimeForMonth:
    payment_interval: ApiMonth
    duration_in_seconds: int


@attr.s(auto_attribs=True)
class ApiDatastoreToken:
    token: str


@attr.s(auto_attribs=True)
class ApiLoggedTimeGroupedByMonth:
    logged_time: List[ApiLoggedTimeForMonth]


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


@attr.s(auto_attribs=True)
class ApiAnnotation:
    id: str
    typ: str
    owner: ApiUser
    name: str
    description: str
    state: str
    tracing_time: Optional[int]  # millis
    modified: int
    data_store: ApiDataStore


@attr.s(auto_attribs=True)
class ApiFolderWithParent:
    id: str
    name: str
    parent: Optional[str]
