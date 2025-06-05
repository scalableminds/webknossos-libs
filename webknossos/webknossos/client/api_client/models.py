from typing import Any, Literal

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
class ApiTeamAdd:
    name: str


@attr.s(auto_attribs=True)
class ApiBoundingBox:
    top_left: tuple[int, int, int]
    width: int
    height: int
    depth: int


@attr.s(auto_attribs=True)
class ApiAdditionalAxis:
    name: str
    bounds: tuple[int, int]
    index: int


@attr.s(auto_attribs=True)
class ApiDataLayer:
    name: str
    category: str
    element_class: str
    bounding_box: ApiBoundingBox
    resolutions: list[tuple[int, int, int]]
    additional_axes: list[ApiAdditionalAxis] | None = None
    largest_segment_id: int | None = None
    default_view_configuration: dict[str, Any] | None = None


@attr.s(auto_attribs=True)
class ApiVoxelSize:
    unit: str
    factor: tuple[float, float, float]


@attr.s(auto_attribs=True)
class ApiDataSource:
    data_layers: list[ApiDataLayer] | None = None
    status: str | None = None
    scale: ApiVoxelSize | None = None


@attr.s(auto_attribs=True)
class ApiMetadata:
    key: str
    type: str
    value: Any


@attr.s(auto_attribs=True)
class ApiDataset:
    id: str
    name: str
    is_public: bool
    folder_id: str
    allowed_teams: list[ApiTeam]
    tags: list[str]
    directory_name: str
    owning_organization: str
    data_store: ApiDataStore
    data_source: ApiDataSource
    created: float
    metadata: list[ApiMetadata] | None = None
    description: str | None = None


@attr.s(auto_attribs=True)
class ApiDatasetId:
    id: str
    name: str
    organization: str
    directory_name: str


@attr.s(auto_attribs=True)
class ApiDatasetExploreAndAddRemote:
    remote_uri: str
    dataset_name: str
    folder_path: str | None = None
    data_store_name: str | None = None


@attr.s(auto_attribs=True)
class ApiDatasetAnnounceUpload:
    dataset_name: str
    organization: str
    initial_team_ids: list[str]
    folder_id: str | None
    require_unique_name: bool


@attr.s(auto_attribs=True)
class ApiDatasetIsValidNewNameResponse:
    is_valid: bool
    errors: list[str] | None = None


@attr.s(auto_attribs=True)
class ApiSharingToken:
    sharing_token: str


@attr.s(auto_attribs=True)
class ApiDatasetUploadInformation:
    upload_id: str


@attr.s(auto_attribs=True)
class ApiDatasetUploadSuccess:
    new_dataset_id: str


@attr.s(auto_attribs=True)
class ApiDatasetManualUploadSuccess:
    new_dataset_id: str
    directory_name: str


@attr.s(auto_attribs=True)
class ApiLinkedLayerIdentifier:
    organization_id: str
    data_set_name: str
    layer_name: str
    new_layer_name: str | None = None


@attr.s(auto_attribs=True)
class ApiReserveDatasetUploadInformation:
    upload_id: str
    name: str
    organization: str
    total_file_count: int
    total_file_size_in_bytes: int
    initial_teams: list[str]
    layers_to_link: list[ApiLinkedLayerIdentifier] | None = None
    folder_id: str | None = None


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
    settings: dict[str, Any] | None = None
    tracing_type: Literal["skeleton", "volume", "hybrid"] | None = None


@attr.s(auto_attribs=True)
class ApiTaskTypeCreate:
    summary: str
    description: str
    team_id: str
    team_name: str
    tracing_type: Literal["skeleton", "volume", "hybrid"]
    settings: dict[str, Any] | None = None


@attr.s(auto_attribs=True)
class ApiExperience:
    domain: str
    value: int


@attr.s(auto_attribs=True)
class ApiScript:
    id: str
    name: str
    owner: str


@attr.s(auto_attribs=True)
class ApiTask:
    id: str
    project_id: str
    dataset_id: str
    status: ApiTaskStatus
    type: ApiTaskType
    needed_experience: ApiExperience
    bounding_box: ApiBoundingBox | None
    edit_position: tuple[int, int, int]
    edit_rotation: tuple[float, float, float]
    script: ApiScript | None = None


@attr.s(auto_attribs=True)
class ApiNmlTaskParameters:
    task_type_id: str
    needed_experience: ApiExperience
    pending_instances: int
    project_name: str
    script_id: str | None
    bounding_box: ApiBoundingBox | None


@attr.s(auto_attribs=True)
class ApiTaskParameters:
    task_type_id: str
    needed_experience: ApiExperience
    pending_instances: int
    project_name: str
    script_id: str | None
    bounding_box: ApiBoundingBox | None
    dataset_id: str
    edit_position: tuple[int, int, int]
    edit_rotation: tuple[float, float, float]


@attr.s(auto_attribs=True)
class ApiSingleTaskCreationResult:
    success: ApiTask | None = None
    error: str | None = None


@attr.s(auto_attribs=True)
class ApiTaskCreationResult:
    tasks: list[ApiSingleTaskCreationResult]
    warnings: list[str]


@attr.s(auto_attribs=True)
class ApiTeamMembership:
    id: str
    name: str
    is_team_manager: bool


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
    teams: list[ApiTeamMembership]
    experiences: dict[str, int]


@attr.s(auto_attribs=True)
class ApiUserCompact:
    id: str
    first_name: str
    last_name: str
    email: str | None = None
    is_admin: bool | None = None
    is_dataset_manager: bool | None = None


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
    logged_time: list[ApiLoggedTimeForMonth]


@attr.s(auto_attribs=True)
class ApiProject:
    id: str
    name: str
    team: str
    team_name: str
    priority: int
    paused: bool
    owner: ApiUserCompact | None  # None in case you have no read access on the owner
    is_blacklisted_from_report: bool
    expected_time: int | None = None


@attr.s(auto_attribs=True)
class ApiProjectCreate:
    name: str
    team: str
    priority: int
    paused: bool
    is_blacklisted_from_report: bool
    expected_time: int | None = None
    owner: str | None = None


@attr.s(auto_attribs=True)
class ApiAnnotation:
    id: str
    typ: str
    owner: ApiUserCompact
    name: str
    description: str
    state: str
    modified: int
    data_store: ApiDataStore | None = None
    tracing_time: int | None = None  # millis


@attr.s(auto_attribs=True)
class ApiAnnotationIdentifier:
    id: str
    typ: str


@attr.s(auto_attribs=True)
class ApiAnnotationUploadResult:
    annotation: ApiAnnotationIdentifier
    messages: list[str]


@attr.s(auto_attribs=True)
class ApiFolderWithParent:
    id: str
    name: str
    parent: str | None = None


@attr.s(auto_attribs=True)
class ApiFolder:
    id: str
    name: str
    allowed_teams: list[ApiTeam]
    allowed_teams_cumulative: list[ApiTeam]
    is_editable: bool
    metadata: list[ApiMetadata] | None = None
