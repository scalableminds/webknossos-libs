"""WEBKNOSSOS annotation module for working with skeleton and volume data.

This module provides the Annotation class for handling WEBKNOSSOS annotations, which can
contain two types of data:

    1. Skeleton Data: Represented by the `Skeleton` class, e.g. for annotating neural pathways
       and structures.
    2. Volume Annotation Layers: Used for segmentation data, can be exported as
       `SegmentationLayer` objects.

The module supports various operations including:
    * Loading/saving annotations from/to .nml or .zip files
    * Downloading/uploading annotations from/to WEBKNOSSOS
    * Working with skeleton data and volume layers
    * Exporting volume layers to datasets

Typical usage Examples:
    ```python
    # Download an annotation from WEBKNOSSOS
    annotation = Annotation.download("annotation_id")

    # Add a volume layer with fallback for efficient processing
    annotation.add_volume_layer(
        name="segmentation",
        fallback_layer="original_segmentation"
    )

    # Save the annotation locally
    annotation.save("my_annotation.zip")
    ```

Notes:
    For volume annotations, using fallback layers is recommended for better performance
    in WEBKNOSSOS. Adding volume annotation data programmatically should be avoided
    when possible.

See Also:
    * Skeleton documentation: /webknossos/skeleton_annotation/index.html
    * Volume annotation documentation: /webknossos/volume_annotation/index.html
"""

import json
import logging
import re
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from enum import Enum, unique
from io import BytesIO
from os import PathLike
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import BinaryIO, Union, cast, overload
from zipfile import ZIP_DEFLATED, ZipFile
from zlib import Z_BEST_SPEED

import attr
from cluster_tools.executor_protocol import Executor
from upath import UPath
from zipp import Path as ZipPath

import webknossos._nml as wknml

from ..client.api_client.models import ApiAnnotation
from ..dataset import (
    SEGMENTATION_CATEGORY,
    DataFormat,
    Dataset,
    Layer,
    RemoteDataset,
    SegmentationLayer,
)
from ..dataset.defaults import PROPERTIES_FILE_NAME, SSL_CONTEXT
from ..dataset.properties import DatasetProperties, dataset_converter
from ..geometry import NDBoundingBox, Vec3Int
from ..skeleton import Skeleton
from ..utils import get_executor_for_args, time_since_epoch_in_ms
from ._nml_conversion import annotation_to_nml, nml_to_skeleton

logger = logging.getLogger(__name__)

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]


MAG_RE = r"((\d+-\d+-)?\d+)"
SEP_RE = r"(\/|\\)"
CUBE_RE = rf"z\d+{SEP_RE}y\d+{SEP_RE}x\d+\.wkw"
ANNOTATION_WKW_PATH_RE = re.compile(rf"{MAG_RE}{SEP_RE}(header\.wkw|{CUBE_RE})")


@attr.define
class SegmentInformation:
    name: str | None
    anchor_position: Vec3Int | None
    color: Vector4 | None
    metadata: dict[str, str | int | float | Sequence[str]]


@attr.define
class _VolumeLayer:
    id: int
    name: str
    fallback_layer_name: str | None
    data_format: DataFormat
    zip: ZipPath | None
    segments: dict[int, SegmentInformation]
    largest_segment_id: int | None

    def _default_zip_name(self) -> str:
        return f"data_{self.id}_{self.name}.zip"


def _extract_zip_folder(zip_file: ZipFile, out_path: Path, prefix: str) -> None:
    for zip_entry in zip_file.filelist:
        if zip_entry.filename.startswith(prefix) and not zip_entry.is_dir():
            out_file_path = out_path / (zip_entry.filename[len(prefix) :])
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                zip_file.open(zip_entry, "r") as zip_f,
                out_file_path.open("wb") as out_f,
            ):
                copyfileobj(zip_f, out_f)


@attr.define
class Annotation:
    """Represents an annotation from WEBKNOSSOS containing skeleton and/or volume data.

    The Annotation class provides functionality to:

    1. Load/save annotations from/to .nml or .zip files
    2. Download/upload annotations from/to WEBKNOSSOS
    3. Work with skeleton data and volume layers
    4. Export volume layers to datasets

    Attributes:
        name: Name of the annotation.
        skeleton: Skeleton object containing tree/node data.
        owner_name: Name of annotation owner.
        annotation_id: Unique identifier for the annotation.
        time: Creation timestamp in ms since epoch.
        edit_position: 3D position coordinates.
        edit_rotation: 3D rotation angles.
        zoom_level: Current zoom level.
        metadata: Dictionary of custom metadata.
        task_bounding_box: Optional bounding box for task annotations.
        user_bounding_boxes: List of user-defined bounding boxes.

    Examples:
        Create a new annotation:
        ```python
        ann = Annotation(
            name="my_annotation",
            dataset_name="sample_dataset",
            voxel_size=(11.2, 11.2, 25.0)
        )
        ```

        Load annotation from file:
        ```
        ann = Annotation.load("annotation.nml")
        ```

        Download from WEBKNOSSOS:
        ```
        ann = Annotation.download("annotation_id")
        ```

        Save annotation:
        ```
        ann.save("annotation.zip")
        ```

        Add volume layer:
        ```
        ann.add_volume_layer(
            name="segmentation",
            fallback_layer="segmentation_layer"
        )
        ```
    """

    name: str
    skeleton: Skeleton = None  # type: ignore[assignment]
    # The following underscored attributes are just for initialization
    # in case the skeleton is not given. They are always None as attributes.
    _dataset_name: str | None = None
    _voxel_size: Vector3 | None = None
    _organization_id: str | None = None
    _description: str | None = None
    owner_name: str | None = None
    annotation_id: str | None = None
    time: int | None = attr.ib(factory=time_since_epoch_in_ms)
    edit_position: Vector3 | None = None
    edit_rotation: Vector3 | None = None
    zoom_level: float | None = None
    metadata: dict[str, str] = attr.Factory(dict)
    task_bounding_box: NDBoundingBox | None = None
    user_bounding_boxes: list[NDBoundingBox] = attr.Factory(list)
    _volume_layers: list[_VolumeLayer] = attr.field(factory=list, init=False)

    @classmethod
    def _set_init_docstring(cls) -> None:
        Annotation.__init__.__doc__ = """
        Initializes a new Annotation instance.

        To initialize a local annotation, please provide the `name` argument, and either
        the `skeleton` argument, or a `dataset_name` and `voxel_size`.
        When supplying `skeleton` passing `dataset_name`, `voxel_size`, `organization_id` or
        `description` is not allowed as the attributes of the skeleton are used in this case.

        Args:
            name: The name of the annotation.
            skeleton: Optional Skeleton instance. If provided, must not specify dataset_name/voxel_size.
            _dataset_name: Required if skeleton not provided. Name of the dataset.
            _voxel_size: Required if skeleton not provided. Tuple of (x, y, z) voxel dimensions.
            _organization_id: Optional organization ID if skeleton not provided.
            _description: Optional description if skeleton not provided.
            owner_name: Optional name of the annotation owner.
            annotation_id: Optional unique identifier.
            time: Optional timestamp in ms since epoch.
            edit_position: Optional tuple of (x, y, z) view position.
            edit_rotation: Optional tuple of (x, y, z) view rotation angles.
            zoom_level: Optional view zoom level.
            metadata: Optional dictionary of custom metadata.
            task_bounding_box: Optional bounding box for task annotations.
            user_bounding_boxes: Optional list of user-defined bounding boxes.

        Raises:
            AssertionError: If neither skeleton nor dataset_name/voxel_size are provided,
                            or if skeleton is provided along with dataset attributes.
        """

    def __attrs_post_init__(self) -> None:
        if self.skeleton is None:
            assert self._dataset_name is not None, (
                "Please either supply a skeleton or dataset_name for Annotation()."
            )
            assert self._voxel_size is not None, (
                "Please supply a voxel_size for Annotation()."
            )
            self.skeleton = Skeleton(
                dataset_name=self._dataset_name,
                voxel_size=self._voxel_size,
                organization_id=self._organization_id,
                description=self._description,
            )
            self._dataset_name = None
            self._voxel_size = None
            self._organization_id = None
            self._description = None
        else:
            assert all(
                i is None
                for i in [
                    self._dataset_name,
                    self._voxel_size,
                    self._organization_id,
                    self._description,
                ]
            ), (
                "When supplying a skeleton for Annotation(), passing dataset_name, voxel_size, organization_id or description is not allowed. "
                + "The attributes of the skeleton are used in this case."
            )

    @property
    def dataset_name(self) -> str:
        """Name of the dataset this annotation belongs to.

        Proxies to skeleton.dataset_name.
        """
        return self.skeleton.dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name: str) -> None:
        self.skeleton.dataset_name = dataset_name

    @property
    def dataset_id(self) -> str | None:
        """ID of the dataset this annotation belongs to.

        Proxies to skeleton.dataset_id.
        """
        return self.skeleton.dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id: str | None) -> None:
        self.skeleton.dataset_id = dataset_id

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Voxel dimensions in nanometers (x, y, z).

        Proxies to skeleton.voxel_size.
        """
        return self.skeleton.voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size: tuple[float, float, float]) -> None:
        self.skeleton.voxel_size = voxel_size

    @property
    def organization_id(self) -> str | None:
        """ID of the organization owning this annotation.

        Proxies to skeleton.organization_id.
        """
        return self.skeleton.organization_id

    @organization_id.setter
    def organization_id(self, organization_id: str | None) -> None:
        self.skeleton.organization_id = organization_id

    @property
    def description(self) -> str | None:
        """Optional description of the annotation.

        Proxies to skeleton.description.
        """
        return self.skeleton.description

    @description.setter
    def description(self, description: str | None) -> None:
        self.skeleton.description = description

    @classmethod
    def load(cls, annotation_path: str | PathLike) -> "Annotation":
        """Loads an annotation from a file.

        Supports loading from:
        - .nml files (skeleton-only annotations)
        - .zip files (containing .nml and optional volume layers)

        Args:
            annotation_path: Path to the .nml or .zip file.

        Returns:
            Annotation: The loaded annotation instance.

        Raises:
            AssertionError: If the file doesn't exist or has invalid extension.
            RuntimeError: If the file format is invalid.

        Examples:
            ```python
            # Load from NML
            ann = Annotation.load("annotation.nml")

            # Load from ZIP
            ann = Annotation.load("annotation.zip")
            ```
        """
        annotation_path = Path(annotation_path)
        assert annotation_path.exists(), (
            f"Annotation path {annotation_path} does not exist."
        )
        if annotation_path.suffix == ".zip":
            return cls._load_from_zip(annotation_path)
        elif annotation_path.suffix == ".nml":
            with annotation_path.open(mode="rb") as f:
                return cls._load_from_nml(annotation_path.stem, f)
        else:
            raise RuntimeError(
                "The loaded annotation must have the suffix .zip or .nml, but is {annotation_path.suffix}"
            )

    @overload
    @classmethod
    def download(
        cls,
        annotation_id_or_url: str,
        webknossos_url: str | None = None,
        *,
        skip_volume_data: bool = False,
    ) -> "Annotation": ...

    @overload
    @classmethod
    def download(
        cls,
        annotation_id_or_url: str,
        webknossos_url: str | None = None,
        *,
        skip_volume_data: bool = False,
        _return_context: bool,
    ) -> tuple["Annotation", AbstractContextManager[None]]: ...

    @classmethod
    def download(
        cls,
        annotation_id_or_url: str,
        webknossos_url: str | None = None,
        *,
        skip_volume_data: bool = False,
        retry_count: int = 5,
        _return_context: bool = False,
    ) -> Union["Annotation", tuple["Annotation", AbstractContextManager[None]]]:
        """Downloads an annotation from WEBKNOSSOS.

        Args:
            annotation_id_or_url: Either an annotation ID or complete WEBKNOSSOS URL.
                Example URL: https://webknossos.org/annotations/[id]
            webknossos_url: Optional custom WEBKNOSSOS instance URL.
            skip_volume_data: If True, omits downloading volume layer data.
            _return_context: Internal use only.

        Returns:
            Annotation: The downloaded annotation instance.

        Examples:
            ```python
            # Download by ID
            ann = Annotation.download("5f7d3a...")

            # Download by URL
            ann = Annotation.download("https://webknossos.org/annotations/5f7d3a...")

            # Skip volume data
            ann = Annotation.download("5f7d3a...", skip_volume_data=True)
            ```
        """
        from ..client._resolve_short_link import resolve_short_link
        from ..client.context import _get_api_client, _get_context, webknossos_context

        annotation_id_or_url = resolve_short_link(annotation_id_or_url)

        match = re.match(_ANNOTATION_URL_REGEX, annotation_id_or_url)
        if match is not None:
            assert webknossos_url is None, (
                "When Annotation.download() is be called with an annotation url, "
                + "e.g. Annotation.download('https://webknossos.org/annotations/6114d9410100009f0096c640'), "
                + "annotation_type and webknossos_url must not be set."
            )
            annotation_id = match.group("annotation_id")
            webknossos_url = match.group("webknossos_url")
        else:
            annotation_id = annotation_id_or_url

        if webknossos_url is not None:
            webknossos_url = webknossos_url.rstrip("/")
        if webknossos_url is not None and webknossos_url != _get_context().url:
            warnings.warn(
                f"[INFO] The supplied url {webknossos_url} does not match your current context {_get_context().url}. "
                + "Using no token, only public annotations can be downloaded. "
                + "Please see https://docs.webknossos.org/api/webknossos/client/context.html to adapt the URL and token."
            )
            context: AbstractContextManager[None] = webknossos_context(
                webknossos_url, token=None
            )
        else:
            context = nullcontext()

        with context:
            client = _get_api_client()
            file_body, filename = client.annotation_download(
                annotation_id,
                skip_volume_data=skip_volume_data,
                retry_count=retry_count,
            )

        if filename.endswith(".nml"):
            annotation = Annotation._load_from_nml(filename[:-4], BytesIO(file_body))
        else:
            assert filename.endswith(".zip"), (
                f"Downloaded annotation should have the suffix .zip or .nml, but has filename {filename}"
            )
            annotation = Annotation._load_from_zip(BytesIO(file_body))

        if _return_context:
            return annotation, context
        else:
            return annotation

    @classmethod
    def open_remote(
        cls,
        annotation_id_or_url: str,
        webknossos_url: str | None = None,
    ) -> "RemoteAnnotation":
        (
            annotation,
            context,
        ) = Annotation.download(
            annotation_id_or_url=annotation_id_or_url,
            webknossos_url=webknossos_url,
            skip_volume_data=True,
            _return_context=True,
        )
        assert annotation.annotation_id is not None
        assert annotation.organization_id is not None
        assert annotation.owner_name is not None

        with context:
            return RemoteAnnotation(
                annotation.annotation_id,
                annotation.organization_id,
                annotation.dataset_id,
                name=annotation.name,
                skeleton=annotation.skeleton,
                owner_name=annotation.owner_name,
                time=annotation.time,
                edit_position=annotation.edit_position,
                edit_rotation=annotation.edit_rotation,
                zoom_level=annotation.zoom_level,
                metadata=annotation.metadata,
                task_bounding_box=annotation.task_bounding_box,
                user_bounding_boxes=annotation.user_bounding_boxes,
            )

    @classmethod
    def open_as_remote_dataset(
        cls,
        annotation_id_or_url: str,
        webknossos_url: str | None = None,
    ) -> Dataset:
        """Opens an annotation directly as a remote dataset from WEBKNOSSOS.

        This is a convenience method that combines downloading an annotation and converting it
        to a remote dataset in one step. It's useful when you want to work with the annotation
        data as a dataset without storing it locally.

        Args:
            annotation_id_or_url: Either an annotation ID or complete WEBKNOSSOS URL.
                Example URL: https://webknossos.org/annotations/[id]
            webknossos_url: Optional custom WEBKNOSSOS instance URL.

        Returns:
            Dataset: A remote dataset instance representing the annotation.

        Examples:
            ```python
            # Open by ID
            dataset = Annotation.open_as_remote_dataset("5f7d3a...")

            # Open by URL
            dataset = Annotation.open_as_remote_dataset(
                "https://webknossos.org/annotations/5f7d3a..."
            )

            # Access layers
            layer = dataset.get_layer("segmentation")
            ```

        Notes:
            This method automatically skips downloading volume data locally for efficiency,
            as the data will be streamed from the remote source.
        """
        (
            annotation,
            context,
        ) = Annotation.download(
            annotation_id_or_url=annotation_id_or_url,
            webknossos_url=webknossos_url,
            skip_volume_data=True,
            _return_context=True,
        )

        with context:
            return annotation.get_remote_annotation_dataset()

    @classmethod
    def _load_from_nml(
        cls,
        name: str,
        nml_content: BinaryIO,
        possible_volume_paths: list[ZipPath] | None = None,
    ) -> "Annotation":
        nml = wknml.Nml.parse(nml_content)

        annotation = cls(
            name=name,
            skeleton=nml_to_skeleton(nml),
            owner_name=nml.get_meta("username"),
            annotation_id=nml.get_meta("annotationId"),
            time=nml.parameters.time,
            edit_position=nml.parameters.editPosition,
            edit_rotation=nml.parameters.editRotation,
            zoom_level=nml.parameters.zoomLevel,
            task_bounding_box=nml.parameters.taskBoundingBox,
            user_bounding_boxes=nml.parameters.userBoundingBoxes or [],
            metadata={
                i.name: i.content
                for i in nml.meta
                if i.name not in ["username", "annotationId"]
            },
        )
        annotation._volume_layers = cls._parse_volumes(nml, possible_volume_paths)
        return annotation

    @staticmethod
    def _parse_volumes(
        nml: wknml.Nml, possible_paths: list[ZipPath] | None
    ) -> list[_VolumeLayer]:
        volume_layers = []
        layers_with_not_found_location = []
        layers_without_location = []
        for volume in nml.volumes:
            if possible_paths is None:  # when parsing NML files
                volume_path = None
                if volume.location is not None:
                    # This should only happen if a zipped nml
                    # is unpacked and loaded directly.
                    layers_with_not_found_location.append(volume)
                    volume_path = None
            elif volume.location is None:
                volume_path = None
                layers_without_location.append(volume)
            else:
                fitting_volume_paths = [
                    i for i in possible_paths if str(i.at) == volume.location
                ]
                if len(fitting_volume_paths) == 1:
                    with fitting_volume_paths[0].open(mode="rb") as f:
                        with ZipFile(f) as volume_layer_zipfile:
                            if len(volume_layer_zipfile.filelist) == 0:
                                volume_path = None
                            else:
                                volume_path = fitting_volume_paths[0]
                else:
                    layers_with_not_found_location.append(volume)
                    volume_path = None

            segments = {}
            if volume.segments is not None:
                for segment in volume.segments:
                    segments[segment.id] = SegmentInformation(
                        name=segment.name,
                        anchor_position=segment.anchor_position,
                        color=segment.color,
                        metadata={i.key: i.value for i in segment.metadata},
                    )
            volume_layers.append(
                _VolumeLayer(
                    id=volume.id,
                    name="Volume" if volume.name is None else volume.name,
                    fallback_layer_name=volume.fallback_layer,
                    data_format=(
                        DataFormat(volume.format)
                        if volume.format is not None
                        else DataFormat.WKW
                    ),
                    zip=volume_path,
                    segments=segments,
                    largest_segment_id=volume.largest_segment_id,
                )
            )
        assert len(set(i.id for i in volume_layers)) == len(volume_layers), (
            "Some volume layers have the same id, this is not allowed."
        )
        if len(layers_without_location) > 0:
            warnings.warn(
                "[INFO] Omitting the volume layer annotation data for layers "
                + f"{[v.name or v.id for v in layers_without_location]}, "
                + "as their location is not referenced in the NML."
            )
        if len(layers_with_not_found_location) > 0:
            warnings.warn(
                "[INFO] Omitting the volume layer annotation data for layers "
                + f"{[v.name or v.id for v in layers_without_location]}, "
                + f"as their referenced files {[v.location for v in layers_without_location]} "
                + "cannot be found."
            )
        return volume_layers

    @classmethod
    def _load_from_zip(cls, content: str | PathLike | BinaryIO) -> "Annotation":
        zipfile = ZipFile(content)
        paths = [ZipPath(zipfile, i.filename) for i in zipfile.filelist]
        nml_paths = [i for i in paths if i.suffix == ".nml"]
        assert len(nml_paths) > 0, "Couldn't find an nml file in the supplied zip-file."
        assert len(nml_paths) == 1, (
            f"There must be exactly one nml file in the zip-file, but found {len(nml_paths)}."
        )
        with nml_paths[0].open(mode="rb") as f:
            return cls._load_from_nml(nml_paths[0].stem, f, possible_volume_paths=paths)

    def save(self, path: str | PathLike) -> None:
        """Saves the annotation to a file.

        For skeleton-only annotations, saves as .nml file.
        For annotations with volume layers, saves as .zip file containing .nml and layers.

        Args:
            path: Target path ending in .nml or .zip
                    (.zip required if annotation contains volume layers)

        Raises:
            AssertionError: If path has invalid extension or trying to save
                            volume layers to .nml file.

        Examples:
            ```
            # Save skeleton-only annotation
            annotation.save("skeleton.nml")

            # Save with volume layers
            annotation.save("full_annotation.zip")
            ```
        """
        path = Path(path)
        assert path.suffix in [
            ".zip",
            ".nml",
        ], "The target path must have a .zip or .nml suffix."

        if path.suffix == ".zip":
            with ZipFile(
                path, mode="x", compression=ZIP_DEFLATED, compresslevel=Z_BEST_SPEED
            ) as zipfile:
                self._write_to_zip(zipfile)
        else:
            assert len(self._volume_layers) == 0, (
                f"Annotation {self.name} contains volume annotations and cannot be saved as an NML file. "
                + "Please use a .zip path instead."
            )
            nml = annotation_to_nml(self)
            with open(path, "wb") as f:
                nml.write(f)

    def merge_fallback_layer(
        self,
        target: Path,
        dataset_directory: Path,
        volume_layer_name: str | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Merges volume annotations with their fallback layer.

        Creates a new dataset containing the merged result of volume annotations
        and fallback layer data.

        Args:
            target: Output path for merged dataset.
            dataset_directory: Directory containing the fallback dataset.
            volume_layer_name: Name of volume layer to merge if multiple exist.
            executor: Optional executor for parallel processing.

        Raises:
            AssertionError: If no volume layers exist.
            AssertionError: If specified volume layer doesn't exist.

        Examples:
            ```python
            # Merge annotations with fallback
            annotation.merge_fallback_layer(
                Path("merged_dataset"),
                Path("original_dataset")
            )
            ```
        """
        annotation_volumes = list(self.get_volume_layer_names())

        output_dataset = Dataset(
            target,
            voxel_size=self.voxel_size,
        )
        assert len(annotation_volumes) > 0, (
            "Annotation does not contain any volume layers!"
        )

        if volume_layer_name is not None:
            assert volume_layer_name in annotation_volumes, (
                f'Volume layer name "{volume_layer_name}" not found in annotation'
            )
        else:
            assert len(annotation_volumes) == 1, (
                "Volume layer name was not provided and more than one volume layer found in annotation"
            )
            volume_layer_name = annotation_volumes[0]

        volume_layer = self._get_volume_layer(volume_layer_name=volume_layer_name)
        fallback_layer_name = volume_layer.fallback_layer_name

        if fallback_layer_name is None:
            logger.info("No fallback layer found, save annotation as dataset.")
            self.export_volume_layer_to_dataset(output_dataset)

        else:
            fallback_dataset_path = dataset_directory / self.dataset_name
            fallback_layer = Dataset.open(fallback_dataset_path).get_layer(
                fallback_layer_name
            )

            if volume_layer.zip is None:
                logger.info("No volume annotation found. Copy fallback layer.")
                with get_executor_for_args(args=None, executor=executor) as executor:
                    output_dataset.add_layer_as_copy(
                        fallback_layer, compress=True, executor=executor
                    )

            else:
                tmp_annotation_layer_name = f"{self.name}-TMP"
                logger.info(
                    "Unpack annotation layer %s temporarily in %s as %s",
                    volume_layer_name,
                    output_dataset.name,
                    tmp_annotation_layer_name,
                )
                # NOTE(erjel): Cannot use "temporary_volume_layer_copy" here, since tmp folders
                # might not be accessible from slurm compute nodes.
                input_annotation_layer = self.export_volume_layer_to_dataset(
                    output_dataset,
                    layer_name=tmp_annotation_layer_name,
                    volume_layer_name=volume_layer_name,
                )

                input_annotation_mag = input_annotation_layer.get_finest_mag()
                fallback_mag = fallback_layer.get_mag(input_annotation_mag.mag)

                logger.info(
                    "Create layer %s in %s",
                    fallback_layer.name,
                    output_dataset.path,
                )
                output_layer = output_dataset.add_layer_like(
                    fallback_layer, fallback_layer.name
                )

                with get_executor_for_args(args=None, executor=executor) as executor:
                    logger.info(
                        "Copy Mag %s from %s to %s",
                        fallback_mag.mag,
                        fallback_layer.path,
                        output_layer.path,
                    )
                    output_mag = output_layer.add_mag_as_copy(
                        fallback_mag,
                        compress=True,
                        executor=executor,
                    )

                    output_mag.merge_with_view(input_annotation_mag, executor)

                logger.info("Delete temporary annotation layer")
                output_dataset.delete_layer(tmp_annotation_layer_name)
                logger.info("Done.")

    def upload(self) -> str:
        """Uploads the annotation to WEBKNOSSOS.

        Uses the current webknossos_context for authentication and target instance.
        See webknossos.webknossos_context() for configuration.

        Returns:
            str: URL of the uploaded annotation in WEBKNOSSOS.

        Raises:
            RuntimeError: If no valid authentication is configured.

        Examples:
            ```python
            with webknossos.webknossos_context(token="my_token"):
                url = annotation.upload()
                print(f"Uploaded to: {url}")
            ```
        """
        from ..client.context import _get_api_client, _get_context

        context = _get_context()

        if self.organization_id is None:
            if context.organization_id is None:
                raise RuntimeError(
                    "No organization_id specified in the annotation or context. "
                    + "Either set it for this annotation `my_annotation.organization_id = 'org_id'` "
                    + "or wrap the upload in a context:\n"
                    + "`with wk.webknossos_context(organization_id='org_id'):`\n"
                    + "`    my_annotation.upload()`"
                )
            self.organization_id = context.organization_id

        client = _get_api_client(enforce_auth=True)
        response_annotation_info = client.annotation_upload(
            self._binary_zip(), f"{self.name}.zip", createGroupForEachFile=False
        )

        return f"{context.url}/annotations/{response_annotation_info.annotation.typ}/{response_annotation_info.annotation.id}"

    def _binary_zip(self) -> bytes:
        with BytesIO() as buffer:
            with ZipFile(
                buffer, mode="a", compression=ZIP_DEFLATED, compresslevel=Z_BEST_SPEED
            ) as zipfile:
                self._write_to_zip(zipfile)
            return buffer.getvalue()

    def _write_to_zip(self, zipfile: ZipFile) -> None:
        nml = annotation_to_nml(self)
        with BytesIO() as buffer:
            nml.write(buffer)
            nml_str = buffer.getvalue().decode("utf-8")
        zipfile.writestr(self.name + ".nml", nml_str)

        for volume_layer in self._volume_layers:
            if volume_layer.zip is None:
                with BytesIO() as buffer:
                    with ZipFile(buffer, mode="a"):
                        pass
                    layer_content = buffer.getvalue()
            else:
                layer_content = volume_layer.zip.read_bytes()
            zipfile.writestr(
                volume_layer._default_zip_name(),
                layer_content,
            )

    def get_remote_annotation_dataset(self) -> Dataset:
        """Returns a streamed dataset of the annotation from WEBKNOSSOS.

        Creates a remote dataset that includes fallback layers and potentially any active agglomerate mappings.
        Requires the annotation to be already stored in WEBKNOSSOS.

        Returns:
            Dataset: Remote dataset instance representing the annotation.

        Raises:
            ValueError: If annotation_id is not set (annotation not in WEBKNOSSOS).

        Examples:
            ```python
            # Stream annotation as dataset
            dataset = annotation.get_remote_annotation_dataset()

            # Access layers
            layer = dataset.get_layer("segmentation")
            ```
        Notes:
            After an agglomerate mapping was activated in WEBKNOSSOS, it is applied to this method as soon
            as the first volume editing action is done. Note that this behavior might change
            in the future.
        """

        from ..client.context import _get_context

        if self.annotation_id is None:
            raise ValueError(
                "The annotation_id is not set, cannot get the corresponding dataset."
            )

        context = _get_context()
        token: str | None
        if self.organization_id is None:
            token = context.required_token
            organization_id = context.organization_id
        else:
            token = context.token
            organization_id = self.organization_id
            # avoid requiring authentication
            if token is not None:
                if organization_id != context.organization_id:
                    warnings.warn(
                        "[WARNING] The annotation used with `get_remote_annotation_dataset` "
                        + "specifies a different organization id than the current context. "
                        + f"The annotation uses {organization_id}, the context {context.organization_id}.",
                        UserWarning,
                    )
        if self.dataset_id is None:
            dataset_id = context.api_client.dataset_id_from_name(
                self.dataset_name, organization_id
            )
        else:
            dataset_id = self.dataset_id

        dataset_info = context.api_client.dataset_info(dataset_id)

        datastore_url = dataset_info.data_store.url
        url_prefix = context.get_datastore_api_client(datastore_url).url_prefix

        zarr_path = UPath(
            f"{url_prefix}/annotations/zarr/{self.annotation_id}/",
            headers={} if token is None else {"X-Auth-Token": token},
            ssl=SSL_CONTEXT,
        )
        return Dataset.open(zarr_path)

    def get_remote_base_dataset(
        self,
        sharing_token: str | None = None,
        webknossos_url: str | None = None,
    ) -> RemoteDataset:
        """Returns a remote dataset connection to the base dataset.

        Creates a connection to the dataset referenced by this annotation.

        Args:
            sharing_token: Optional token for accessing private datasets.
            webknossos_url: Optional custom WEBKNOSSOS instance URL.

        Returns:
            RemoteDataset: Connection to the base dataset.

        Examples:
            ```python
            # Connect to base dataset
            dataset = annotation.get_remote_base_dataset()

            # With sharing token
            dataset = annotation.get_remote_base_dataset(
                sharing_token="abc123"
            )
            ```
        """
        return Dataset.open_remote(
            self.dataset_name,
            self.organization_id,
            sharing_token=sharing_token,
            webknossos_url=webknossos_url,
            dataset_id=self.dataset_id,
        )

    def get_volume_layer_names(self) -> Iterable[str]:
        """Returns names of all volume layers in the annotation.

        Returns:
            Iterable[str]: Iterator of volume layer names.

        Examples:
            ```python
            # Print all layer names
            for name in annotation.get_volume_layer_names():
                print(f"Found layer: {name}")
            ```
        """
        return (i.name for i in self._volume_layers)

    def add_volume_layer(
        self,
        name: str,
        fallback_layer: Layer | str | None = None,
        volume_layer_id: int | None = None,
    ) -> None:
        """Adds a new volume layer to the annotation.

        Volume layers can be used to store segmentation data. Using fallback layers
        is recommended for better performance in WEBKNOSSOS.

        Args:
            name: Name of the volume layer.
            fallback_layer: Optional reference to existing segmentation layer in WEBKNOSSOS.
                          Can be Layer instance or layer name.
            volume_layer_id: Optional explicit ID for the layer.
                           Auto-generated if not provided.

        Raises:
            AssertionError: If volume_layer_id already exists.
            AssertionError: If fallback_layer is provided but not a segmentation layer.

        Examples:
            ```python
            # Add basic layer
            annotation.add_volume_layer("segmentation")

            # Add with fallback
            annotation.add_volume_layer("segmentation", fallback_layer="base_segmentation")
            ```
        """

        if volume_layer_id is None:
            volume_layer_id = max((i.id for i in self._volume_layers), default=-1) + 1
        else:
            assert volume_layer_id not in [i.id for i in self._volume_layers], (
                f"volume layer id {volume_layer_id} already exists in annotation {self.name}."
            )
        fallback_layer_name: str | None
        if isinstance(fallback_layer, Layer):
            assert fallback_layer.category == SEGMENTATION_CATEGORY, (
                "The fallback layer must be a segmentation layer."
            )
            fallback_layer_name = fallback_layer.name
        elif fallback_layer is not None:
            fallback_layer_name = str(fallback_layer)
        else:
            fallback_layer_name = None
        self._volume_layers.append(
            _VolumeLayer(
                id=volume_layer_id,
                name=name,
                fallback_layer_name=fallback_layer_name,
                data_format=DataFormat.Zarr3,
                zip=None,
                segments={},
                largest_segment_id=None,
            )
        )

    def _get_volume_layer(
        self,
        volume_layer_name: str | None = None,
        volume_layer_id: int | None = None,
    ) -> _VolumeLayer:
        assert len(self._volume_layers) > 0, "No volume annotations present."

        if len(self._volume_layers) == 1:
            volume_layer = self._volume_layers[0]
            if volume_layer_id is not None and volume_layer_id != volume_layer.id:
                warnings.warn(
                    f"[INFO] Only a single volume annotation is present and its id {volume_layer.id} does not fit the given id {volume_layer_id}."
                )
            if (
                volume_layer_name is not None
                and volume_layer.name is not None
                and volume_layer_name != volume_layer.name
            ):
                warnings.warn(
                    f"[INFO] Only a single volume annotation is present and its name {volume_layer.name} "
                    + f"does not fit the given name {volume_layer_name}."
                )
            return volume_layer

        if volume_layer_id is not None:
            for volume_layer in self._volume_layers:
                if volume_layer_id == volume_layer.id:
                    if (
                        volume_layer_name is not None
                        and volume_layer.name is not None
                        and volume_layer_name != volume_layer.name
                    ):
                        warnings.warn(
                            f"[WARNING] The volume annotation was matched by id {volume_layer_id}, "
                            + f"but its name {volume_layer.name} does not fit the given name {volume_layer_name}."
                        )
                    return volume_layer
            available_ids = [volume_layer.id for volume_layer in self._volume_layers]
            raise ValueError(
                f"Couldn't find a volume annotation with the id {volume_layer_id}, available are {available_ids}."
            )
        elif volume_layer_name is not None:
            fitting_volume_layers = [
                i for i in self._volume_layers if i.name == volume_layer_name
            ]
            assert len(fitting_volume_layers) != 0, (
                f"The specified volume name {volume_layer_name} could not be found in this annotation."
            )
            assert len(fitting_volume_layers) == 1, (
                f"More than one volume annotation has the name {volume_layer_name}. "
                + "Please specify the exact annotation via the volume_layer_id argument. "
                + f"The matching annotations have the ids {[i.id for i in fitting_volume_layers]}"
            )
            return fitting_volume_layers[0]
        else:
            raise ValueError(
                "The annotation contains multiple volume layers. "
                + "Please specify which layer should be used via volume_layer_name or volume_layer_id."
            )

    def delete_volume_layer(
        self,
        volume_layer_name: str | None = None,
        volume_layer_id: int | None = None,
    ) -> None:
        """Removes a volume layer from the annotation.

        Args:
            volume_layer_name: Name of the layer to delete if multiple exist.
            volume_layer_id: ID of the layer to delete if multiple exist.

        Raises:
            ValueError: If neither name nor ID is provided when multiple layers exist.
            AssertionError: If specified layer doesn't exist.

        Examples:
            ```python
            # Delete by name
            annotation.delete_volume_layer("unused_layer")

            # Delete by ID
            annotation.delete_volume_layer(volume_layer_id=2)
            ```
        """
        layer_id = self._get_volume_layer(
            volume_layer_name=volume_layer_name,
            volume_layer_id=volume_layer_id,
        ).id
        self._volume_layers = [i for i in self._volume_layers if i.id != layer_id]

    def export_volume_layer_to_dataset(
        self,
        dataset: Dataset,
        layer_name: str = "volume_layer",
        volume_layer_name: str | None = None,
        volume_layer_id: int | None = None,
    ) -> SegmentationLayer:
        """Exports a volume layer to a dataset.

        Creates a new layer in the target dataset containing the volume annotation data.

        Args:
            dataset: Target Dataset instance.
            layer_name: Name for the new layer (default: "volume_layer").
            volume_layer_name: Name of source volume layer if multiple exist.
            volume_layer_id: ID of source volume layer if multiple exist.

        Returns:
            SegmentationLayer: The created layer in the target dataset.

        Raises:
            AssertionError: If specified volume layer doesn't exist.
            AssertionError: If volume layer data is not available.

        Examples:
            ```python
            # Export to dataset
            layer = annotation.export_volume_layer_to_dataset(
                dataset,
                layer_name="exported_segmentation"
            )
            ```
        """

        volume_layer = self._get_volume_layer(
            volume_layer_name=volume_layer_name,
            volume_layer_id=volume_layer_id,
        )
        volume_zip_path = volume_layer.zip

        largest_segment_id = volume_layer.largest_segment_id

        assert volume_zip_path is not None, (
            "The selected volume layer data is not available and cannot be exported."
        )

        with volume_zip_path.open(mode="rb") as f:
            data_zip = ZipFile(f)
            if volume_layer.data_format == DataFormat.WKW:
                wrong_files = [
                    i.filename
                    for i in data_zip.filelist
                    if ANNOTATION_WKW_PATH_RE.search(i.filename) is None
                ]
                assert len(wrong_files) == 0, (
                    f"The annotation contains unexpected files: {wrong_files}"
                )
                data_zip.extractall(dataset.path / layer_name)
                layer = cast(
                    SegmentationLayer,
                    dataset.add_layer_for_existing_files(
                        layer_name,
                        category=SEGMENTATION_CATEGORY,
                        largest_segment_id=largest_segment_id,
                    ),
                )
            elif volume_layer.data_format == DataFormat.Zarr3:
                datasource_properties = dataset_converter.structure(
                    json.loads(data_zip.read(PROPERTIES_FILE_NAME)), DatasetProperties
                )
                assert len(datasource_properties.data_layers) == 1, (
                    f"Volume data zip must contain exactly one layer, got {len(datasource_properties.data_layers)}"
                )
                layer_properties = datasource_properties.data_layers[0]
                internal_layer_name = layer_properties.name
                layer_properties.name = layer_name

                _extract_zip_folder(
                    data_zip, dataset.path / layer_name, f"{internal_layer_name}/"
                )

                layer = cast(
                    SegmentationLayer,
                    dataset._add_existing_layer(layer_properties),
                )

        best_mag_view = layer.get_finest_mag()

        if largest_segment_id is None:
            max_value = max(
                (
                    view.read().max()
                    for view in best_mag_view.get_views_on_disk(read_only=True)
                ),
                default=0,
            )
            layer.largest_segment_id = int(max_value)
        else:
            layer.largest_segment_id = largest_segment_id
        return layer

    @contextmanager
    def temporary_volume_layer_copy(
        self,
        volume_layer_name: str | None = None,
        volume_layer_id: int | None = None,
        read_only: bool = True,
    ) -> Iterator[SegmentationLayer]:
        """Creates a temporary copy of a volume layer as a dataset.

        Context manager that provides temporary access to volume layer data
        as a SegmentationLayer.

        Args:
            volume_layer_name: Name of target layer if multiple exist.
            volume_layer_id: ID of target layer if multiple exist.
            read_only: If True, prevents modifications to the layer.

        Yields:
            SegmentationLayer: Temporary layer containing volume data.

        Examples:
            ```python
            # Temporarily access volume data
            with annotation.temporary_volume_layer_copy("segmentation") as layer:
                data = layer.get_mag(1).read()
            ```
        """

        with TemporaryDirectory() as tmp_annotation_dir:
            input_annotation_dataset = Dataset(
                tmp_annotation_dir,
                name="tmp_annotation_dataset",
                voxel_size=self.voxel_size,
                exist_ok=True,
            )

            input_annotation_layer = self.export_volume_layer_to_dataset(
                input_annotation_dataset,
                "volume_layer",
                volume_layer_name=volume_layer_name,
                volume_layer_id=volume_layer_id,
            )

            input_annotation_dataset._read_only = read_only

            yield input_annotation_layer

    def get_volume_layer_segments(
        self,
        volume_layer_name: str | None = None,
        volume_layer_id: int | None = None,
    ) -> dict[int, SegmentInformation]:
        """Returns segment information for a volume layer.

        Returns a mutable dictionary mapping segment IDs to their metadata.
        Changes to the returned dictionary are reflected in the annotation locally.

        Args:
            volume_layer_name: Name of the target volume layer if multiple exist.
            volume_layer_id: ID of the target volume layer if multiple exist.

        Returns:
            dict[int, SegmentInformation]: Dictionary mapping segment IDs to their information.

        Raises:
            ValueError: If neither name nor ID is provided when multiple layers exist.
            AssertionError: If specified layer doesn't exist.

        Examples:
            ```python
            # Get segments for a layer
            segments = annotation.get_volume_layer_segments("segmentation_layer")

            # Update segment name
            segments[1].name = "Cell A"
            ```
        Notes:
            Any changes performed on the online version of the annotaiton in webknossos are not
            synced automatically. The annotation needs to be re-downloaded to update segment information.
        """

        layer = self._get_volume_layer(
            volume_layer_name=volume_layer_name,
            volume_layer_id=volume_layer_id,
        )
        return layer.segments


Annotation._set_init_docstring()


@unique
class AnnotationType(Enum):
    """Annotations can be of different types which has to be specified when using `Annotation.download()`
    with an annotation id."""

    EXPLORATIONAL = "Explorational"
    """**Explorational** annotations are all annotations created without the task system, e.g.
    by uploading an annotation or using the "Create Annotation" Button in the dataset view in webknossos."""
    TASK = "Task"
    """The **Task** type is automatically assigned to all annotations that are instances of a task.
    See also `Task`."""
    _COMPOUND_TASK = "CompoundTask"
    """A collection of annotations of all instances of a task. Currently not supported,
    please download all annotations individually instead, see `Task`."""
    _COMPOUND_PROJECT = "CompoundProject"
    """A collection of annotations of all instances of all tasks in a project. Currently not supported,
    please download all annotations individually instead, see `Project`."""
    _COMPOUND_TASK_TYPE = "CompoundTaskType"
    """A collection of annotations of all instances of all tasks in a project. Currently not supported,
    please download all annotations individually instead, see `Project`."""


@unique
class AnnotationState(Enum):
    """This Enum contains the state of annotations belonging to tasks.
    Can be retrieved via `Task` instances, getting `AnnotationInfo` from `task.get_annotation_infos()`.
    """

    FINISHED = "Finished"
    ACTIVE = "Active"
    CANCELLED = "Cancelled"
    INITIALIZING = "Initializing"


_COMPOUND_ANNOTATION_TYPES = [
    AnnotationType._COMPOUND_PROJECT,
    AnnotationType._COMPOUND_TASK,
    AnnotationType._COMPOUND_TASK_TYPE,
]

_ANNOTATION_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/annotations/"
    + rf"((?P<annotation_type>{'|'.join(i.value for i in AnnotationType.__members__.values())})/)?"
    + r"(?P<annotation_id>[0-9A-Fa-f]*)"
)


class RemoteAnnotation(Annotation):
    def __init__(
        self,
        annotation_id: str,
        organization_id: str,
        dataset_id: str | None = None,
        *,
        name: str,
        skeleton: Skeleton,
        owner_name: str,
        time: int | None = None,
        edit_position: Vector3 | None = None,
        edit_rotation: Vector3 | None = None,
        zoom_level: float | None = None,
        task_bounding_box: NDBoundingBox | None = None,
        user_bounding_boxes: list[NDBoundingBox] | None = None,
        metadata: dict[str, str] = {},
    ) -> None:
        """A remote Annotation instance.
        Note: Please not initialize this class directly, use Annotation.open_remote() instead."""
        self.annotation_id = annotation_id
        self.skeleton = skeleton
        self.dataset_id = dataset_id
        self.organization_id = organization_id
        self.name = name
        self.owner_name = owner_name
        self.time = time
        self.edit_position = edit_position
        self.edit_rotation = edit_rotation
        self.zoom_level = zoom_level
        self.task_bounding_box = task_bounding_box
        self.user_bounding_boxes = user_bounding_boxes or []
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"RemoteAnnotation({self.url})"

    @property
    def url(self) -> str:
        from webknossos.client.context import _get_context

        return f"{_get_context().url}/annotations/{self.annotation_id}"

    @property
    def name(self) -> str:
        return self._get_annotation_info().name

    @name.setter
    def name(self, value: str) -> None:
        assert isinstance(value, str), "Name must be a string."
        self._set_annotation_info(name=value)

    @property
    def description(self) -> str | None:
        return self._get_annotation_info().description

    @description.setter
    def description(self, value: str) -> None:
        assert isinstance(value, str), "Description must be a string."
        self._set_annotation_info(description=value)

    def _get_annotation_info(self) -> ApiAnnotation:
        from webknossos.client.context import _get_api_client

        client = _get_api_client(True)
        assert self.annotation_id is not None, "Annotation ID must be set."
        return client.annotation_info(self.annotation_id)

    def _set_annotation_info(
        self, name: str | None = None, description: str | None = None
    ) -> None:
        from webknossos.client.context import _get_api_client

        client = _get_api_client(True)
        annotation_info = self._get_annotation_info()
        name = name if name is not None else annotation_info.name
        description = (
            description if description is not None else annotation_info.description
        )
        assert self.annotation_id is not None, "Annotation ID must be set."
        client.annotation_edit(
            annotation_typ=annotation_info.typ,
            annotation_id=self.annotation_id,
            annotation=ApiAnnotation(
                id=self.annotation_id,
                typ=annotation_info.typ,
                owner=annotation_info.owner,
                name=name,
                description=description,
                state=annotation_info.state,
                modified=annotation_info.modified,
                data_store=annotation_info.data_store,
                tracing_time=annotation_info.tracing_time,
            ),
        )

    def save(self, path: str | PathLike) -> None:
        raise NotImplementedError(
            "Remote annotations cannot be saved. Changes are applied ."
        )
