from typing import Any, Dict, List, Optional, Tuple

import attr

# Request and response bodies for wk/datastore routes
# Should contain only the fields that are actually used by the python libs
# Note that keys are converted to/from camelCase to match this
# Optional fields in response bodies should always have “= None” defaults.
# When adding things here, check which fields are optional on the WK server side


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
    bounding_box: ApiBoundingBox
    resolutions: List[Tuple[int, int, int]]
    largest_segment_id: Optional[int] = None
    default_view_configuration: Optional[Dict[str, Any]] = None


@attr.s(auto_attribs=True)
class ApiDataSource:
    data_layers: Optional[List[ApiDataLayer]] = None
    status: Optional[str] = None
    scale: Optional[Tuple[float, float, float]] = None


@attr.s(auto_attribs=True)
class ApiDataset:
    name: str
    is_public: bool
    folder_id: str
    allowed_teams: List[ApiTeam]
    tags: List[str]
    data_store: ApiDataStore
    data_source: ApiDataSource
    display_name: Optional[str] = None
    description: Optional[str] = None


@attr.s(auto_attribs=True)
class ApiSharingToken:
    sharing_token: str


@attr.s(auto_attribs=True)
class ApiDatasetUploadInformation:
    upload_id: str


@attr.s(auto_attribs=True)
class ApiLinkedLayerIdentifier:
    organization_name: str
    data_set_name: str
    layer_name: str
    new_layer_name: Optional[str] = None


@attr.s(auto_attribs=True)
class ApiReserveDatasetUploadInformation:
    upload_id: str
    name: str
    organization: str
    total_file_count: int
    initial_teams: List[str]
    layers_to_link: Optional[List[ApiLinkedLayerIdentifier]] = None
    folder_id: Optional[str] = None


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
class ApiExperience:
    domain: str
    value: int


@attr.s(auto_attribs=True)
class ApiNmlTaskParameters:
    task_type_id: str
    needed_experience: ApiExperience
    pending_instances: int
    project_name: str
    script_id: Optional[str]
    bounding_box: Optional[ApiBoundingBox]


@attr.s(auto_attribs=True)
class ApiTaskParameters:
    task_type_id: str
    needed_experience: ApiExperience
    pending_instances: int
    project_name: str
    script_id: Optional[str]
    bounding_box: Optional[ApiBoundingBox]
    data_set: str
    edit_position: Tuple[int, int, int]
    edit_rotation: Tuple[float, float, float]


@attr.s(auto_attribs=True)
class ApiSingleTaskCreationResult:
    error: Optional[str]
    success: Optional[ApiTask]


@attr.s(auto_attribs=True)
class ApiTaskCreationResult:
    tasks: List[ApiSingleTaskCreationResult]
    warnings: List[str]


@attr.s(auto_attribs=True)
class ApiTeamMembership:
    id: str
    name: str


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
    experiences: Dict[str, int]


@attr.s(auto_attribs=True)
class ApiUserCompact:
    id: str
    email: str
    first_name: str
    last_name: str
    is_admin: bool
    is_dataset_manager: bool


@attr.s(auto_attribs=True)
class ApiMonth:
    month: int
    year: int


@attr.s(auto_attribs=True)
class ApiLoggedTimeForMonth:
    payment_interval: ApiMonth
    duration_in_seconds: int


@attr.s(auto_attribs=True)
class ApiDataStoreToken:
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
    owner: ApiUserCompact
    priority: int
    paused: bool
    expected_time: Optional[int] = None


@attr.s(auto_attribs=True)
class ApiAnnotation:
    id: str
    typ: str
    owner: ApiUserCompact
    name: str
    description: str
    state: str
    modified: int
    data_store: ApiDataStore
    tracing_time: Optional[int] = None  # millis


@attr.s(auto_attribs=True)
class ApiAnnotationIdentifier:
    id: str
    typ: str


@attr.s(auto_attribs=True)
class ApiAnnotationUploadResult:
    annotation: ApiAnnotationIdentifier
    messages: List[str]


@attr.s(auto_attribs=True)
class ApiFolderWithParent:
    id: str
    name: str
    parent: Optional[str] = None
