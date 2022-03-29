"""Annotations can contain annotated data in two forms:
  - [skeleton data](/webknossos/skeleton_annotation.html), as provided by the `Skeleton` class, and
  - [volume annotation layers](/webknossos/volume_annotation.html) (or volume layers short),
    which can be exported as a `SegmentationLayer`, see `export_volume_layer_to_dataset()`
    and `temporary_volume_layer_copy()`.

Usually, annotations should be created manually in the webKnossos interface and can be downloaded using
`Annotation.download()`. The downloaded instance is not persisted to disk automatically, please use `save()`
for this purpose. The general purpose file format is `.zip` files containing an `.nml` file with
meta-information and the skeleton data and also containing inner `.zip` files for the volume layers.
For skeleton-only annotations without volume layers `.nml` files can be used directly. Both formats
are compatible with the webKnossos up- and downloads.

To prepare volume annotations in the code for correction of segmentation data in the webKnossos interface,
please use `add_volume_layer()` with the `fallback_layer` argument, referencing a segmentation layer that
is available on webKnossos (e.g. using the `Dataset` upload before).
Correcting segmentations using fallback layers is much more efficient, adding volume
annotation data programmatically is discouraged therefore.
"""

import cgi
import re
import warnings
from contextlib import contextmanager, nullcontext
from enum import Enum, unique
from io import BytesIO
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    BinaryIO,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from zipfile import ZIP_DEFLATED, ZipFile
from zlib import Z_BEST_SPEED

import attr
import httpx
from zipp import Path as ZipPath

import webknossos._nml as wknml
from webknossos.annotation._nml_conversion import annotation_to_nml, nml_to_skeleton
from webknossos.dataset import SEGMENTATION_CATEGORY, Dataset, Layer, SegmentationLayer
from webknossos.geometry import BoundingBox
from webknossos.skeleton import Skeleton
from webknossos.utils import time_since_epoch_in_ms, warn_deprecated

Vector3 = Tuple[float, float, float]


MAG_RE = r"((\d+-\d+-)?\d+)"
SEP_RE = r"(\/|\\)"
CUBE_RE = fr"z\d+{SEP_RE}y\d+{SEP_RE}x\d+\.wkw"
ANNOTATION_WKW_PATH_RE = re.compile(fr"{MAG_RE}{SEP_RE}(header\.wkw|{CUBE_RE})")


@attr.define
class _VolumeLayer:
    id: int
    name: str
    fallback_layer_name: Optional[str]
    zip: Optional[ZipPath]

    def _default_zip_name(self) -> str:
        return f"data_{self.id}_{self.name}.zip"


@attr.define
class Annotation:
    name: str
    skeleton: Skeleton = None  # type: ignore[assignment]
    # The following underscored attributes are just for initialization
    # in case the skeleton is not given. They are always None as attributes.
    _dataset_name: Optional[str] = None
    _scale: Optional[Vector3] = None
    _organization_id: Optional[str] = None
    _description: Optional[str] = None
    username: Optional[str] = None
    annotation_id: Optional[str] = None
    time: Optional[int] = attr.ib(factory=time_since_epoch_in_ms)
    edit_position: Optional[Vector3] = None
    edit_rotation: Optional[Vector3] = None
    zoom_level: Optional[float] = None
    metadata: Dict[str, str] = attr.Factory(dict)
    task_bounding_box: Optional[BoundingBox] = None
    user_bounding_boxes: List[BoundingBox] = attr.Factory(list)
    _volume_layers: List[_VolumeLayer] = attr.field(factory=list, init=False)

    @classmethod
    def _set_init_docstring(cls) -> None:
        Annotation.__init__.__doc__ = """
        To initialize a local annotation, please provide the `name` argument, and either
        the `skeleton` argument, or a `dataset_name` and `scale`.
        When supplying `skeleton` passing `dataset_name`, `scale`, `organization_id` or
        `description` is not allowed as the attributes of the skeleton are used in this case.
        """

    def __attrs_post_init__(self) -> None:
        if self.skeleton is None:
            assert (
                self._dataset_name is not None
            ), "Please either supply a skeleton or dataset_name for Annotation()."
            assert self._scale is not None, "Please supply a scale for Annotation()."
            self.skeleton = Skeleton(
                dataset_name=self._dataset_name,
                scale=self._scale,
                organization_id=self._organization_id,
                description=self._description,
            )
            self._dataset_name = None
            self._scale = None
            self._organization_id = None
            self._description = None
        else:
            assert all(
                i is None
                for i in [
                    self._dataset_name,
                    self._scale,
                    self._organization_id,
                    self._description,
                ]
            ), (
                "When supplying a skeleton for Annotation(), passing dataset_name, scale, organization_id or description is not allowed. "
                + "The attributes of the skeleton are used in this case."
            )

    @property
    def dataset_name(self) -> str:
        """This attribute is a proxy for `skeleton.dataset_name`."""
        return self.skeleton.dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name: str) -> None:
        self.skeleton.dataset_name = dataset_name

    @property
    def scale(self) -> Tuple[float, float, float]:
        """This attribute is a proxy for `skeleton.scale`."""
        return self.skeleton.scale

    @scale.setter
    def scale(self, scale: Tuple[float, float, float]) -> None:
        self.skeleton.scale = scale

    @property
    def organization_id(self) -> Optional[str]:
        """This attribute is a proxy for `skeleton.organization_id`."""
        return self.skeleton.organization_id

    @organization_id.setter
    def organization_id(self, organization_id: Optional[str]) -> None:
        self.skeleton.organization_id = organization_id

    @property
    def description(self) -> Optional[str]:
        """This attribute is a proxy for `skeleton.description`."""
        return self.skeleton.description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        self.skeleton.description = description

    @classmethod
    def load(cls, annotation_path: Union[str, PathLike]) -> "Annotation":
        """Loads a `.nml` file or a `.zip` file containing an NML and possibly also volume
        layers. Returns the `Annotation` object."""
        annotation_path = Path(annotation_path)
        assert (
            annotation_path.exists()
        ), f"Annotation path {annotation_path} does not exist."
        if annotation_path.suffix == ".zip":
            return cls._load_from_zip(annotation_path)
        elif annotation_path.suffix == ".nml":
            with annotation_path.open(mode="rb") as f:
                annotation, nml = cls._load_from_nml(annotation_path.stem, f)
            if len(nml.volumes) > 0:
                warnings.warn(
                    "The loaded nml contains references to volume layer annotations. "
                    + "Those can only be loaded from a zip-file containing the nml and the volume annotation layer zips. "
                    + "Omitting the volume layer annotations."
                )
            return annotation
        else:
            raise RuntimeError(
                "The loaded annotation must have the suffix .zip or .nml, but is {annotation_path.suffix}"
            )

    @classmethod
    def download(
        cls,
        annotation_id_or_url: str,
        annotation_type: Union[str, "AnnotationType", None] = None,
        webknossos_url: Optional[str] = None,
    ) -> "Annotation":
        """
        * `annotation_id_or_url` may be an annotation id or a full URL to an annotation, e.g.
          `https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640`
        * `annotation_type` must be supplied iff an annotation id was used in the previous argument
        * `webknossos_url` may be supplied if an annotation id was used
          and allows to specifiy in which webknossos instance to search for the annotation.
          It defaults to the url from your current `webknossos_context`, using https://webknossos.org as a fallback.
        """
        from webknossos.client._generated.api.default import annotation_download
        from webknossos.client.context import (
            _get_context,
            _get_generated_client,
            webknossos_context,
        )

        match = re.match(_ANNOTATION_URL_REGEX, annotation_id_or_url)
        if match is not None:
            assert webknossos_url is None and annotation_type is None, (
                "When Annotation.download() is be called with an annotation url, "
                + "e.g. Annotation.download('https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640'), "
                + "annotation_type and webknossos_url must not be set."
            )
            annotation_id = match.group("annotation_id")
            annotation_type = match.group("annotation_type")
            webknossos_url = match.group("webknossos_url")
        else:
            assert annotation_type is not None, (
                "When calling Annotation.download() with an id you must supply the argument annotation_type, "
                + "e.g. 'Task' or 'Explorational'. Alternatively, you can use the full annotation url, "
                + "e.g. Annotation.download('https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640')."
            )
            annotation_id = annotation_id_or_url
        annotation_type = AnnotationType(annotation_type)
        assert (
            annotation_type not in _COMPOUND_ANNOTATION_TYPES
        ), f"Currently compund annotation types are not supported, got {annotation_type}"

        if webknossos_url is not None and webknossos_url != _get_context().url:
            warnings.warn(
                f"The supplied url {webknossos_url} does not match your current context {_get_context().url}. "
                + "Using no token, only public annotations can be downloaded. "
                + "Please see https://docs.webknossos.org/api/webknossos/client/context.html to adapt the URL and token."
            )
            context: ContextManager[None] = webknossos_context(
                webknossos_url, token=None
            )
        else:
            context = nullcontext()

        with context:
            client = _get_generated_client()
            response = annotation_download.sync_detailed(
                typ=annotation_type.value, id=annotation_id, client=client
            )
        assert response.status_code == 200, response
        content_disposition_header = response.headers.get("content-disposition", "")
        _header_value, header_params = cgi.parse_header(content_disposition_header)
        filename = header_params.get("filename", "")
        if filename.endswith(".nml"):
            annotation, nml = Annotation._load_from_nml(
                filename[:-4], BytesIO(response.content)
            )
            assert (
                len(nml.volumes) == 0
            ), "The downloaded NML contains volume tags, it should have downloaded a zip instead."
            return annotation
        else:
            assert filename.endswith(
                ".zip"
            ), f"Downloaded annoation should have the suffix .zip or .nml, but has filename {filename}"
            return Annotation._load_from_zip(BytesIO(response.content))

    @classmethod
    def _load_from_nml(
        cls, name: str, nml_content: BinaryIO
    ) -> Tuple["Annotation", wknml.Nml]:
        nml = wknml.Nml.parse(nml_content)

        return (
            cls(
                name=name,
                skeleton=nml_to_skeleton(nml),
                username=nml.get_meta("username"),
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
            ),
            nml,
        )

    @classmethod
    def _load_from_zip(cls, content: Union[str, PathLike, BinaryIO]) -> "Annotation":
        zipfile = ZipFile(content)
        paths = [ZipPath(zipfile, i.filename) for i in zipfile.filelist]
        nml_paths = [i for i in paths if i.suffix == ".nml"]
        assert len(nml_paths) > 0, "Couldn't find an nml file in the supplied zip-file."
        assert (
            len(nml_paths) == 1
        ), f"There must be exactly one nml file in the zip-file, buf found {len(nml_paths)}."
        with nml_paths[0].open(mode="rb") as f:
            annotation, nml = cls._load_from_nml(nml_paths[0].stem, f)
        volume_layers = []
        for volume in nml.volumes:
            fitting_volume_paths = [i for i in paths if str(i.at) == volume.location]
            assert (
                len(fitting_volume_paths) == 1
            ), f"Couldn't find the file {volume.location} for the volume annotation {volume.name or volume.id}"
            with fitting_volume_paths[0].open(mode="rb") as f:
                with ZipFile(f) as volume_layer_zipfile:
                    if len(volume_layer_zipfile.filelist) == 0:
                        volume_path = None
                    else:
                        volume_path = fitting_volume_paths[0]
            volume_layers.append(
                _VolumeLayer(
                    id=volume.id,
                    name="Volume" if volume.name is None else volume.name,
                    fallback_layer_name=volume.fallback_layer,
                    zip=volume_path,
                )
            )
        assert len(set(i.id for i in volume_layers)) == len(
            volume_layers
        ), "Some volume layers have the same id, this is not allowed."
        annotation._volume_layers = volume_layers
        return annotation

    def save(self, path: Union[str, PathLike]) -> None:
        """
        Stores the annotation as a zip or nml at the given path.
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

    def upload(self) -> str:
        """Uploads the annotation to your current `webknossos_context`."""
        from webknossos.client.context import _get_generated_client

        client = _get_generated_client(enforce_auth=True)
        url = f"{client.base_url}/api/annotations/upload"

        response = httpx.post(
            url=url,
            headers=client.get_headers(),
            cookies=client.get_cookies(),
            timeout=client.get_timeout(),
            data={"createGroupForEachFile": False},
            files={
                f"{self.name}.zip": (f"{self.name}.zip", self._binary_zip()),
            },
        )
        assert (
            response.status_code == 200
        ), f"Failed to upload annotation {self.name}: {response.status_code}: {response.text}"
        response_annotation_info = response.json()["annotation"]
        return f"{client.base_url}/annotations/{response_annotation_info['typ']}/{response_annotation_info['id']}"

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

    def get_volume_layer_names(self) -> Iterable[str]:
        return (i.name for i in self._volume_layers)

    def add_volume_layer(
        self,
        name: str,
        fallback_layer: Union[Layer, str, None] = None,
        volume_layer_id: Optional[int] = None,
    ) -> None:
        """
        Adds a volume layer to the annotation, without manual annotations but possibly referring to
        segmentation data using the `fallback_layer`.
        To prepare volume annotations in the code for correction of segmentation data in the webKnossos interface,
        please use the `fallback_layer` argument, referencing a segmentation layer that is available on webKnossos
        (e.g. using the `Dataset` upload before).
        Correcting segmentations using fallback layers is much more efficient, adding volume annotation data
        programmatically is discouraged therefore."""
        if volume_layer_id is None:
            volume_layer_id = max((i.id for i in self._volume_layers), default=-1) + 1
        else:
            assert volume_layer_id not in [
                i.id for i in self._volume_layers
            ], f"volume layer id {volume_layer_id} already exists in annotation {self.name}."
        fallback_layer_name: Optional[str]
        if isinstance(fallback_layer, Layer):
            assert (
                fallback_layer.category == SEGMENTATION_CATEGORY
            ), "The fallback layer must be a segmentation layer."
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
                zip=None,
            )
        )

    def _get_volume_layer(
        self,
        volume_layer_name: Optional[str] = None,
        volume_layer_id: Optional[int] = None,
    ) -> _VolumeLayer:
        assert len(self._volume_layers) > 0, "No volume annotations present."

        if len(self._volume_layers) == 1:
            volume_layer = self._volume_layers[0]
            if volume_layer_id is not None and volume_layer_id != volume_layer.id:
                warnings.warn(
                    f"Only a single volume annotation is present and its id {volume_layer.id} does not fit the given id {volume_layer_id}."
                )
            if (
                volume_layer_name is not None
                and volume_layer.name is not None
                and volume_layer_name != volume_layer.name
            ):
                warnings.warn(
                    f"Only a single volume annotation is present and its name {volume_layer.name} "
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
                            f"The volume annotation was matched by id {volume_layer_id}, "
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
            assert (
                len(fitting_volume_layers) != 0
            ), f"The specified volume name {volume_layer_name} could not be found in this annotation."
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
        volume_layer_name: Optional[str] = None,
        volume_layer_id: Optional[int] = None,
    ) -> None:
        layer_id = self._get_volume_layer(
            volume_layer_name=volume_layer_name,
            volume_layer_id=volume_layer_id,
        ).id
        self._volume_layers = [i for i in self._volume_layers if i.id != layer_id]

    def export_volume_layer_to_dataset(
        self,
        dataset: Dataset,
        layer_name: str = "volume_layer",
        largest_segment_id: Optional[int] = None,
        volume_layer_name: Optional[str] = None,
        volume_layer_id: Optional[int] = None,
    ) -> SegmentationLayer:
        """
        Given a dataset, this method will export the specified
        volume annotation of this annotation into that dataset
        by creating a new layer.
        The largest_segment_id is computed automatically, unless provided
        explicitly.

        `volume_layer_name` or `volume_layer_id` has to be provided,
        if the annotation contains multiple volume layers.
        Use `get_volume_layer_names()` to look up available layers.
        """
        volume_zip_path = self._get_volume_layer(
            volume_layer_name=volume_layer_name,
            volume_layer_id=volume_layer_id,
        ).zip

        assert (
            volume_zip_path is not None
        ), "The selected volume layer is empty and cannot be exported."

        with volume_zip_path.open(mode="rb") as f:
            data_zip = ZipFile(f)
            wrong_files = [
                i.filename
                for i in data_zip.filelist
                if ANNOTATION_WKW_PATH_RE.search(i.filename) is None
            ]
            assert (
                len(wrong_files) == 0
            ), f"The annotation contains unexpected files: {wrong_files}"
            data_zip.extractall(dataset.path / layer_name)
        layer = cast(
            SegmentationLayer,
            dataset.add_layer_for_existing_files(
                layer_name, category="segmentation", largest_segment_id=0
            ),
        )
        best_mag_view = layer.get_best_mag()

        if largest_segment_id is None:
            max_value = max(
                view.read().max()
                for view in best_mag_view.get_views_on_disk(read_only=True)
            )
            layer.largest_segment_id = int(max_value)
        else:
            layer.largest_segment_id = largest_segment_id
        return layer

    @contextmanager
    def temporary_volume_layer_copy(
        self,
        volume_layer_name: Optional[str] = None,
        volume_layer_id: Optional[int] = None,
    ) -> Iterator[SegmentationLayer]:

        """
        Given a volume annotation path, create a temporary dataset which
        contains the volume annotation. Returns the corresponding `Layer`.

        `volume_layer_name` or `volume_layer_id` has to be provided,
        if the annotation contains multiple volume layers.
        """

        with TemporaryDirectory() as tmp_annotation_dir:
            input_annotation_dataset = Dataset(
                tmp_annotation_dir,
                name="tmp_annotation_dataset",
                scale=self.scale,
                exist_ok=True,
            )

            input_annotation_layer = self.export_volume_layer_to_dataset(
                input_annotation_dataset,
                "volume_layer",
                volume_layer_name=volume_layer_name,
                volume_layer_id=volume_layer_id,
            )

            yield input_annotation_layer


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
    + fr"(?P<annotation_type>{'|'.join(i.value for i in AnnotationType.__members__.values())})/"
    + r"(?P<annotation_id>[0-9A-Fa-f]*)"
)


def open_annotation(annotation_path: Union[str, PathLike]) -> "Annotation":
    """Deprecated."""
    if Path(annotation_path).exists():
        warn_deprecated("open_annotation", "Annotation.load")
        return Annotation.load(annotation_path)
    else:
        assert isinstance(
            annotation_path, str
        ), f"Called open_annotation with a path-like, but {annotation_path} does not exist."
        warn_deprecated("open_annotation", "Annotation.download")
        return Annotation.download(annotation_path)
