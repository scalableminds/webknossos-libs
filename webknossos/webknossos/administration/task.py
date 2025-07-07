import logging
from typing import Any, Literal

import attr

from webknossos.geometry.vec3_int import Vec3IntLike

from ..annotation import Annotation, AnnotationInfo
from ..client.api_client.models import (
    ApiBoundingBox,
    ApiExperience,
    ApiNmlTaskParameters,
    ApiTask,
    ApiTaskCreationResult,
    ApiTaskParameters,
    ApiTaskType,
    ApiTaskTypeCreate,
)
from ..client.context import _get_api_client, _get_context
from ..dataset.dataset import RemoteDataset
from ..geometry import BoundingBox, Vec3Int
from ..utils import warn_deprecated
from .project import Project
from .user import Team

logger = logging.getLogger(__name__)


@attr.frozen
class TaskStatus:
    pending_instance_count: int
    active_instance_count: int
    finished_instance_count: int

    @property
    def open_instance_count(self) -> int:
        warn_deprecated("open_instance_count", "pending_instance_count")
        return self.pending_instance_count


@attr.frozen
class TaskExperience:
    """Data class containing information about the experience needed to work on a task"""

    domain: str
    value: int

    @classmethod
    def _from_api_experience(cls, api_experience: ApiExperience) -> "TaskExperience":
        return cls(api_experience.domain, api_experience.value)


@attr.frozen
class TaskType:
    task_type_id: str
    name: str  # Called "summary" in WK UI
    description: str
    team_id: str
    team_name: str
    settings: dict[str, Any] | None = None
    tracingType: str | None = None

    @classmethod
    def _from_api_task_type(cls, api_task_type: ApiTaskType) -> "TaskType":
        return cls(
            api_task_type.id,
            api_task_type.summary,
            api_task_type.description,
            api_task_type.team_id,
            api_task_type.team_name,
        )

    @classmethod
    def get_list(cls) -> list["TaskType"]:
        """
        Retrieve all accessible task types for the current user.

        This class method queries the backend for all task types that the authenticated user
        is authorized to access. It returns a list of `TaskType` instances corresponding to
        the available tasks.

        Returns:
            list[TaskType]: List of all task types accessible to the current user.

        Examples:
            Get all available task types:
                ```
                task_types = TaskType.get_list()
                for task_type in task_types:
                    print(task_type.name)
                ```
        """
        client = _get_api_client(enforce_auth=True)
        api_tasks = client.task_type_list()
        return [cls._from_api_task_type(t) for t in api_tasks]

    @classmethod
    def get_by_id(cls, task_type_id: str) -> "TaskType":
        """
        Retrieve a TaskType instance by its unique ID.

        This class method fetches the task type corresponding to the given `task_type_id`
        from the backend, provided the current user has permission to view it.

        Args:
            task_type_id (str): The unique identifier of the task type to retrieve.

        Returns:
            TaskType: The TaskType instance corresponding to the specified ID.

        Raises:
            UnexpectedStatusError: If the task type cannot be found or the user does not have access.

        Examples:
            Retrieve a task type by ID:
                ```
                task_type = TaskType.get_by_id("1234567890abcdef")
                print(task_type.name)
                ```
        """
        client = _get_api_client(enforce_auth=True)
        return cls._from_api_task_type(client.get_task_type(task_type_id))

    @classmethod
    def get_by_name(cls, name: str) -> "TaskType":
        """
        Get a TaskType by its name.

        Searches for a task type with the specified name among all available task types
        visible to the current user. If found, returns the corresponding TaskType object.

        Args:
            name (str): The name of the task type to retrieve.

        Returns:
            TaskType: The task type object with the specified name.

        Raises:
            ValueError: If no task type with the given name is found.

        Examples:
            ```
            task_type = TaskType.get_by_name("Segmentation")
            print(task_type.id)
            ```
        """
        task_types = cls.get_list()
        for task_type in task_types:
            if task_type.name == name:
                return task_type
        raise ValueError(f"Task type with name '{name}' not found.")

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        team: str | Team,
        tracing_type: Literal["skeleton", "volume", "hybrid"] = "skeleton",
        settings: dict[str, Any] = {
            "mergerMode": False,
            "magRestrictions": {
                "min": 1,
                "max": 1,
            },
            "somaClickingAllowed": True,
            "volumeInterpolationAllowed": False,
            "allowedModes": [],
            "preferredMode": None,
            "branchPointsAllowed": True,
            "clippingDistance": 80,
            "moveValue": 500,
            "displayScalebars": False,
            "newNodeNewTree": False,
            "centerNewNode": True,
            "tdViewDisplayPlanes": "WIREFRAME",
            "tdViewDisplayDatasetBorders": True,
            "tdViewDisplayLayerBorders": False,
            "dynamicSpaceDirection": True,
            "highlightCommentedNodes": False,
            "overrideNodeRadius": True,
            "particleSize": 5,
            "keyboardDelay": 0,
            "displayCrosshair": True,
            "useLegacyBindings": False,
            "fourBit": False,
            "interpolation": True,
            "segmentationOpacity": 0,
            "segmentationPatternOpacity": 40,
            "zoom": 0.8,
            "renderMissingDataBlack": False,
            "loadingStrategy": "BEST_QUALITY_FIRST",
            "clippingDistanceArbitrary": 60,
            "moveValue3d": 600,
            "mouseRotateValue": 0.001,
            "rotateValue": 0.01,
            "sphericalCapRadius": 500,
            "brushSize": 50,
        },
    ) -> "TaskType":
        """
        Creates a new task type and returns it.

        This class method allows you to create a new task type in the system, specifying its name, description, associated team, and tracing type. The created task type is returned as an instance of `TaskType`.

        Args:
            name (str): The name of the new task type. This will be used as the summary for the task type.
            description (str): A detailed description of the task type.
            team (str | Team): The team to which this task type will belong. Can be either a team name (str) or a `Team` object.
            tracing_type (Literal["skeleton", "volume", "hybrid"]): The tracing type for the task. Must be one of "skeleton", "volume", or "hybrid".
            settings (dict[str, Any]): Additional settings for the task type. Information about the task type's configuration options can be found here: https://docs.webknossos.org/webknossos/tasks_projects/tasks.html

        Returns:
            TaskType: The newly created task type object.

        Raises:
            ValueError: If the provided team does not exist or cannot be resolved.
            ApiException: If the API call to create the task type fails.

        Examples:
            Create a new skeleton tracing task type for a team:
                ```
                task_type = TaskType.create(
                    name="Neuron Skeleton Tracing",
                    description="Trace neuron skeletons for connectomics project.",
                    team="NeuroLab",
                    tracing_type="skeleton"
                print(task_type.id)
                ```
        """
        client = _get_api_client(enforce_auth=True)
        if isinstance(team, str):
            team = Team.get_by_name(team)
        team_name = team.name
        team_id = team.id
        api_task_type = client.task_type_create(
            ApiTaskTypeCreate(
                summary=name,
                description=description,
                team_id=team_id,
                team_name=team_name,
                settings=settings,
                tracing_type=tracing_type,
            )
        )
        return cls._from_api_task_type(api_task_type)

    def delete(self) -> None:
        """Deletes the task type."""
        client = _get_api_client(enforce_auth=True)
        client.task_type_delete(self.task_type_id)


@attr.frozen
class Task:
    """Data class containing information about a WEBKNOSSOS task"""

    task_id: str
    project_id: str
    dataset_id: str
    status: TaskStatus
    task_type: TaskType
    experience: TaskExperience
    edit_position: Vec3Int
    edit_rotation: tuple[float, float, float]
    script_id: str | None = None
    bounding_box: BoundingBox | None = None

    @classmethod
    def get_by_id(cls, task_id: str) -> "Task":
        """
        Retrieve a Task by its unique identifier.

        Fetches the task with the specified ID from the backend, provided the current user is authorized to access it.
        This method requires authentication.

        Args:
            task_id (str): The unique identifier of the task to retrieve.

        Returns:
            Task: The Task instance corresponding to the given ID.

        Raises:
            webknossos.client.api_client.errors.UnexpectedStatusError: If the task does not exist or the user is not authorized.

        Examples:
            Get a task by ID:
                ```
                task = Task.get_by_id("task_12345")
                print(task.name)
                ```
        """
        client = _get_api_client(enforce_auth=True)
        api_task = client.task_info(task_id)
        return cls._from_api_task(api_task)

    @classmethod
    def get_list(cls) -> list["Task"]:
        """
        Retrieve all tasks visible to the current user.

        Returns a list of all tasks that the authenticated user is authorized to see. This method queries the backend for all available tasks and returns them as a list of Task objects.

        Returns:
            list[Task]: List of Task objects the user is authorized to access.

        Examples:
            Get all tasks for the current user:
                ```
                tasks = Task.get_list()
                for task in tasks:
                    print(task.name)
                ```
        """
        client = _get_api_client(enforce_auth=True)
        api_tasks = client.task_list()
        return [cls._from_api_task(t) for t in api_tasks]

    @classmethod
    def create_from_annotations(
        cls,
        task_type_id: str | TaskType,
        project_name: str | Project,
        base_annotations: list[Annotation],
        needed_experience_domain: str,
        needed_experience_value: int,
        instances: int = 1,
        script_id: str | None = None,
        bounding_box: BoundingBox | None = None,
    ) -> list["Task"]:
        """
        Create new tasks in WEBKNOSSOS from existing annotations.

        This class method submits one or more tasks to WEBKNOSSOS, using a list of base annotations as input.
        It returns the created Task objects. The method allows specifying the task type, project, required experience,
        number of task instances, and optionally a script and bounding box.

        Args:
            task_type_id: The ID or TaskType object specifying the type of task to create.
            project_name: The name or Project object specifying the project to which the tasks belong.
            base_annotations: List of Annotation objects to use as the basis for the new tasks. Must not be empty.
            needed_experience_domain: The experience domain required for the task (e.g., "proofreading").
            needed_experience_value: The minimum experience value required for the task.
            instances: Number of task instances to create (default: 1).
            script_id: Optional script ID to associate with the task.
            bounding_box: Optional BoundingBox object specifying the spatial extent of the task.

        Returns:
            list[Task]: List of created Task objects.

        Raises:
            AssertionError: If no base annotations are provided.

        Examples:
            ```
            tasks = Task.create_from_annotations(
                task_type_id="proofreading",
                project_name="MyProject",
                base_annotations=[annotation1, annotation2],
                needed_experience_domain="proofreading",
                needed_experience_value=3,
                instances=2,
                script_id="script_123",
                bounding_box=BoundingBox((0, 0, 0), (100, 100, 100))
            for task in tasks:
                print(task.task_id)
            ```

        Note:
            Each annotation in `base_annotations` will be uploaded as a zipped file and associated with the created tasks.
            The method requires authentication and will raise an error if the user is not authenticated.
        """

        assert len(base_annotations) > 0, (
            "Must supply at least one base annotation to create tasks"
        )
        if isinstance(task_type_id, TaskType):
            task_type_id = task_type_id.task_type_id
        if isinstance(project_name, Project):
            project_name = project_name.name

        client = _get_api_client(enforce_auth=True)
        nml_task_parameters = ApiNmlTaskParameters(
            task_type_id=task_type_id,
            needed_experience=ApiExperience(
                domain=needed_experience_domain,
                value=needed_experience_value,
            ),
            pending_instances=instances,
            project_name=project_name,
            script_id=script_id,
            bounding_box=ApiBoundingBox(
                bounding_box.topleft.to_tuple(),
                bounding_box.size.x,
                bounding_box.size.y,
                bounding_box.size.z,
            )
            if bounding_box is not None
            else None,
        )
        annotation_files = [
            (f"{a.name}.zip", a._binary_zip()) for a in base_annotations
        ]
        result = client.tasks_create_from_files(nml_task_parameters, annotation_files)

        return cls._handle_task_creation_result(result)

    @classmethod
    def create(
        cls,
        task_type_id: str | TaskType,
        project_name: str | Project,
        needed_experience_domain: str,
        needed_experience_value: int,
        starting_position: Vec3IntLike,
        dataset_id: str | RemoteDataset,
        dataset_name: str | RemoteDataset | None = None,
        starting_rotation: Vec3IntLike = Vec3Int(0, 0, 0),
        instances: int = 1,
        script_id: str | None = None,
        bounding_box: BoundingBox | None = None,
    ) -> list["Task"]:
        """
        Submits one or more tasks to WEBKNOSSOS based on the specified dataset, starting position, and rotation,
        and returns the created Task objects.

        This method allows you to create annotation or analysis tasks for a given dataset, specifying the required
        experience, starting position, and other parameters. The dataset can be referenced either by its ID or by
        passing a RemoteDataset instance. Optionally, a bounding box can be provided to restrict the task area.

        Args:
            task_type_id: The ID of the task type to create, or a TaskType instance.
            project_name: The name of the project to associate the task with, or a Project instance.
            needed_experience_domain: The experience domain required for the task (e.g., "segmentation").
            needed_experience_value: The minimum experience value required for the task.
            starting_position: The starting position for the task as a Vec3IntLike (x, y, z).
            dataset_id: The dataset ID as a string or a RemoteDataset instance.
            dataset_name: (Deprecated) The dataset name as a string or RemoteDataset instance. Prefer using dataset_id.
            starting_rotation: The starting rotation for the task as a Vec3IntLike (default: Vec3Int(0, 0, 0)).
            instances: The number of task instances to create (default: 1).
            script_id: Optional script ID to associate with the task.
            bounding_box: Optional BoundingBox to restrict the task area.

        Returns:
            list[Task]: A list of created Task objects.

        Raises:
            AssertionError: If neither dataset_id nor dataset_name is provided.
            DeprecationWarning: If dataset_name is used instead of dataset_id.

        Examples:
            Create a new segmentation task for a dataset:
                ```
                tasks = Task.create(
                    task_type_id="segmentation",
                    project_name="MyProject",
                    needed_experience_domain="segmentation",
                    needed_experience_value=10,
                    starting_position=(100, 200, 300),
                    dataset_id="abc123",
                    instances=5
                for task in tasks:
                    print(task.id)
                ```

            Create a task with a bounding box and a RemoteDataset:
                ```
                bbox = BoundingBox((0, 0, 0), (100, 100, 100))
                tasks = Task.create(
                    task_type_id=task_type,
                    project_name=project,
                    needed_experience_domain="proofreading",
                    needed_experience_value=5,
                    starting_position=(0, 0, 0),
                    dataset_id=remote_ds,
                    bounding_box=bbox
                ```
        """

        client = _get_api_client(enforce_auth=True)
        assert dataset_id is not None or dataset_name is not None, (
            "Please provide a dataset_id to create a task."
        )
        starting_position = Vec3Int(starting_position)
        starting_rotation = Vec3Int(starting_rotation)
        if isinstance(task_type_id, TaskType):
            task_type_id = task_type_id.task_type_id
        if isinstance(project_name, Project):
            project_name = project_name.name
        if dataset_id is not None:
            if isinstance(dataset_id, RemoteDataset):
                dataset_id = dataset_id._dataset_id
        else:
            assert dataset_name is not None
            warn_deprecated("dataset_name", "dataset_id")
            if isinstance(dataset_name, RemoteDataset):
                dataset_id = dataset_name._dataset_id
            else:
                context = _get_context()
                dataset_id = client.dataset_id_from_name(
                    dataset_name, context.organization_id
                )

        task_parameters = ApiTaskParameters(
            task_type_id=task_type_id,
            needed_experience=ApiExperience(
                domain=needed_experience_domain,
                value=needed_experience_value,
            ),
            pending_instances=instances,
            project_name=project_name,
            script_id=script_id,
            dataset_id=dataset_id,
            edit_position=starting_position.to_tuple(),
            edit_rotation=starting_rotation.to_tuple(),
            bounding_box=ApiBoundingBox(
                bounding_box.topleft.to_tuple(),
                bounding_box.size.x,
                bounding_box.size.y,
                bounding_box.size.z,
            )
            if bounding_box is not None
            else None,
        )

        response = client.tasks_create([task_parameters])

        return cls._handle_task_creation_result(response)

    @classmethod
    def _from_api_task(cls, api_task: ApiTask) -> "Task":
        return cls(
            api_task.id,
            api_task.project_id,
            api_task.dataset_id,
            TaskStatus(
                api_task.status.pending,
                api_task.status.active,
                api_task.status.finished,
            ),
            TaskType._from_api_task_type(api_task.type),
            TaskExperience._from_api_experience(api_task.needed_experience),
            Vec3Int(api_task.edit_position),
            api_task.edit_rotation,
            api_task.script.id if api_task.script else None,
            BoundingBox.from_tuple2(
                (
                    api_task.bounding_box.top_left,
                    (
                        api_task.bounding_box.width,
                        api_task.bounding_box.height,
                        api_task.bounding_box.depth,
                    ),
                )
            )
            if api_task.bounding_box
            else None,
        )

    def update(
        self,
        remaining_instances: int,
    ) -> "Task":
        """
        Update the task with new parameters.

        Updates the current task instance on the server with the specified number of remaining instances and other task parameters.

        Args:
            remaining_instances: The number of remaining instances for this task.

        Returns:
            Task: The updated Task object as returned by the server.

        Examples:
            ```
            task = Task.get_by_id("task_id")
            updated_task = task.update(remaining_instances=5)
            print(updated_task.remaining_instances)
            ```
        """
        client = _get_api_client(enforce_auth=True)
        api_task = ApiTaskParameters(
            self.task_type.task_type_id,
            ApiExperience(self.experience.domain, self.experience.value),
            remaining_instances,
            self.get_project().name,
            self.script_id,
            ApiBoundingBox(
                self.bounding_box.topleft.to_tuple(),
                self.bounding_box.size.x,
                self.bounding_box.size.y,
                self.bounding_box.size.z,
            )
            if self.bounding_box is not None
            else None,
            self.dataset_id,
            self.edit_position.to_tuple(),
            self.edit_rotation,
        )
        updated = client.task_update(self.task_id, api_task)
        return self._from_api_task(updated)

    def delete(self) -> None:
        """Deletes this task. WARNING: This is irreversible!"""
        client = _get_api_client(enforce_auth=True)
        client.task_delete(self.task_id)

    def get_annotation_infos(self) -> list[AnnotationInfo]:
        """Returns AnnotationInfo objects describing all task instances that have been started by annotators for this task"""
        client = _get_api_client(enforce_auth=True)
        api_annotations = client.annotation_infos_by_task(self.task_id)
        return [AnnotationInfo._from_api_annotation(a) for a in api_annotations]

    def get_project(self) -> Project:
        """Returns the project this task belongs to"""
        return Project.get_by_id(self.project_id)

    @classmethod
    def _handle_task_creation_result(
        cls, result: ApiTaskCreationResult
    ) -> list["Task"]:
        if len(result.warnings) > 0:
            logger.warning(
                f"There were {len(result.warnings)} warnings during task creation:"
            )
        for warning in result.warnings:
            logger.warning(warning)

        successes = []
        errors = []
        for t in result.tasks:
            if t.success is not None:
                successes.append(t.success)
            if t.error is not None:
                errors.append(t.error)
        if len(errors) > 0:
            logger.error(f"{len(errors)} tasks could not be created:")
            for error in errors:
                logger.error(error)
        if len(successes) > 0:
            logger.info(f"{len(successes)} tasks were successfully created.")
        return [cls._from_api_task(t) for t in successes]
