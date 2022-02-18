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
from zipp import Path as ZipPath

import webknossos._nml as wknml
from webknossos.annotation._nml_conversion import annotation_to_nml, nml_to_skeleton
from webknossos.dataset import Dataset, Layer, SegmentationLayer
from webknossos.skeleton import Skeleton
from webknossos.utils import time_since_epoch_in_ms

Vector3 = Tuple[float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]


MAG_RE = r"((\d+-\d+-)?\d+)"
SEP_RE = r"(\/|\\)"
CUBE_RE = fr"z\d+{SEP_RE}y\d+{SEP_RE}x\d+\.wkw"
ANNOTATION_WKW_PATH_RE = re.compile(fr"{MAG_RE}{SEP_RE}(header\.wkw|{CUBE_RE})")


@unique
class AnnotationType(Enum):
    TASK = "Task"
    EXPLORATIONAL = "Explorational"
    COMPOUND_TASK = "CompoundTask"
    COMPOUND_PROJECT = "CompoundProject"
    COMPOUND_TASK_TYPE = "CompoundTaskType"


@unique
class AnnotationState(Enum):
    FINISHED = "Finished"
    ACTIVE = "Active"
    CANCELLED = "Cancelled"
    INITIALIZING = "Initializing"


_COMPOUND_ANNOTATION_TYPES = [
    AnnotationType.COMPOUND_PROJECT,
    AnnotationType.COMPOUND_TASK,
    AnnotationType.COMPOUND_TASK_TYPE,
]

_ANNOTATION_URL_REGEX = re.compile(
    fr"(https?://.*)/annotations/({'|'.join(i.value for i in AnnotationType.__members__.values())})/([0-9A-Fa-f]*)"
)


@attr.define
class _VolumeAnnotation:
    id: int
    name: str
    fallback_layer_name: Optional[str]
    zip: ZipPath

    def _default_zip_name(self) -> str:
        return f"data_{self.id}_{self.name}"


@attr.define
class Annotation:
    name: str
    skeleton: Skeleton = None  # type: ignore[assignment]
    _dataset_name: Optional[
        str
    ] = None  # just for initialization, attribute is always None
    _scale: Optional[
        Vector3
    ] = None  # just for initialization, attribute is always None
    username: Optional[str] = None
    annotation_id: Optional[str] = None
    time: Optional[int] = attr.ib(factory=time_since_epoch_in_ms)
    edit_position: Optional[Vector3] = None
    edit_rotation: Optional[Vector3] = None
    zoom_level: Optional[float] = None
    metadata: Dict[str, str] = attr.Factory(dict)
    task_bounding_box: Optional[IntVector6] = None
    user_bounding_boxes: Optional[List[IntVector6]] = None
    _volume_annotations: List[_VolumeAnnotation] = attr.field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        if self.skeleton is None:
            assert (
                self._dataset_name is not None
            ), "Please either supply a skeleton or dataset_name for Annotation()."
            assert self._scale is not None, "Please supply a scale for Annotation()."
            self.skeleton = Skeleton(dataset_name=self._dataset_name, scale=self._scale)
            self._dataset_name = None
            self._scale = None
        else:
            assert self._dataset_name is None, (
                "When supplying a skeleton for Annotation(), passing dataset_name is not allowed. "
                + "The dataset_name of the skeleton is used."
            )
            assert self._scale is None, (
                "When supplying a skeleton for Annotation(), passing scale is not allowed. "
                + "The scale of the skeleton is used."
            )

    @property
    def dataset_name(self) -> str:
        return self.skeleton.dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name: str) -> None:
        self.skeleton.dataset_name = dataset_name

    @property
    def scale(self) -> Tuple[float, float, float]:
        return self.skeleton.scale

    @scale.setter
    def scale(self, scale: Tuple[float, float, float]) -> None:
        self.skeleton.scale = scale

    @classmethod
    def load(cls, annotation_path: Union[str, PathLike]) -> "Annotation":
        annotation_path = Path(annotation_path)
        assert (
            annotation_path.exists()
        ), f"Annotation path {annotation_path} does not exist."
        if annotation_path.suffix == ".zip":
            return cls._load_from_zip(annotation_path)
        elif annotation_path.suffix == ".nml":
            annotation, nml = cls._load_from_nml(annotation_path)
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
        annotation_type: Union[str, AnnotationType, None] = None,
        webknossos_url: Optional[str] = None,
    ) -> "Annotation":
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
            webknossos_url, annotation_type, annotation_id = match.groups()
        else:
            assert annotation_type is not None, (
                "When calling Annotation.download() with an id you must supply the argument annotation_type, "
                + "e.g. 'Task' or 'Explorational'. Alternatively, you can use the full annotation url, "
                + "e.g. Annotation.download('https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640')."
            )
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
        # TODO allow nml-only download
        return Annotation._load_from_zip(BytesIO(response.content))

    @classmethod
    def _load_from_nml(
        cls, nml_path: Union[Path, ZipPath]
    ) -> Tuple["Annotation", wknml.Nml]:
        with nml_path.open(mode="rb") as f:
            nml = wknml.Nml.parse(f)

        return (
            cls(
                name=nml_path.stem,
                skeleton=nml_to_skeleton(nml),
                username=nml.get_meta("username"),
                annotation_id=nml.get_meta("annotationId"),
                time=nml.parameters.time,
                edit_position=nml.parameters.editPosition,
                edit_rotation=nml.parameters.editRotation,
                zoom_level=nml.parameters.zoomLevel,
                task_bounding_box=nml.parameters.taskBoundingBox,
                user_bounding_boxes=nml.parameters.userBoundingBoxes,
                metadata={i.name: i.content for i in nml.meta},
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
        annotation, nml = cls._load_from_nml(nml_paths[0])
        volume_annotations = []
        for volume in nml.volumes:
            fitting_volume_paths = [i for i in paths if str(i.at) == volume.location]
            assert (
                len(fitting_volume_paths) == 1
            ), f"Couldn't find the file {volume.location} for the volume annotation {volume.name or volume.id}"
            volume_annotations.append(
                _VolumeAnnotation(
                    id=volume.id,
                    name="Volume" if volume.name is None else volume.name,
                    fallback_layer_name=volume.fallback_layer,
                    zip=fitting_volume_paths[0],
                )
            )
        annotation._volume_annotations = volume_annotations
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
            assert len(self._volume_annotations) == 0, (
                f"Annotation {self.name} contains volume annotations and cannot be saved as an NML file. "
                + "Please use a .zip path instead."
            )
            nml = annotation_to_nml(self)
            with open(path, "wb") as f:
                nml.write(f)

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

        for volume_annotation in self._volume_annotations:
            zipfile.writestr(
                volume_annotation._default_zip_name(),
                volume_annotation.zip.read_bytes(),
            )

    def get_volume_annotation_names(self) -> Iterable[str]:
        return (i.name for i in self._volume_annotations)

    # TODO add more methods, e.g. add_volume_annotation

    def _get_volume_annotation(
        self,
        volume_annotation_name: Optional[str],
        volume_annotation_id: Optional[int],
    ) -> _VolumeAnnotation:
        assert len(self._volume_annotations) > 0, "No volume annotations present."

        if len(self._volume_annotations) == 1:
            volume_layer = self._volume_annotations[0]
            if (
                volume_annotation_id is not None
                and volume_annotation_id != volume_layer.id
            ):
                warnings.warn(
                    f"Only a single volume annotation is present and its id {volume_layer.id} does not fit the given id {volume_annotation_id}."
                )
            if (
                volume_annotation_name is not None
                and volume_layer.name is not None
                and volume_annotation_name != volume_layer.name
            ):
                warnings.warn(
                    f"Only a single volume annotation is present and its name {volume_layer.name} "
                    + f"does not fit the given name {volume_annotation_name}."
                )
            return volume_layer

        if volume_annotation_id is not None:
            for volume_layer in self._volume_annotations:
                if volume_annotation_id == volume_layer.id:
                    if (
                        volume_annotation_name is not None
                        and volume_layer.name is not None
                        and volume_annotation_name != volume_layer.name
                    ):
                        warnings.warn(
                            f"The volume annotation was matched by id {volume_annotation_id}, "
                            + f"but its name {volume_layer.name} does not fit the given name {volume_annotation_name}."
                        )
                    return volume_layer
            available_ids = [
                volume_layer.id for volume_layer in self._volume_annotations
            ]
            raise ValueError(
                f"Couldn't find a volume annotation with the id {volume_annotation_id}, available are {available_ids}."
            )
        elif volume_annotation_name is not None:
            fitting_volume_annotations = [
                i for i in self._volume_annotations if i.name == volume_annotation_name
            ]
            assert (
                len(fitting_volume_annotations) != 0
            ), f"The specified volume name {volume_annotation_name} could not be found in this annotation."
            assert len(fitting_volume_annotations) == 1, (
                f"More than one volume annotation has the name {volume_annotation_name}. "
                + "Please specify the exact annotation via the volume_annotation_id argument. "
                + f"The matching annotations have the ids {[i.id for i in fitting_volume_annotations]}"
            )
            return fitting_volume_annotations[0]
        else:
            raise ValueError(
                "The annotation contains multiple volume layers. "
                + "Please specify which layer should be used via volume_annotation_name or volume_annotation_id."
            )

    def save_volume_annotation(
        self,
        dataset: Dataset,
        layer_name: str = "volume_annotation",
        largest_segment_id: Optional[int] = None,
        volume_annotation_name: Optional[str] = None,
        volume_annotation_id: Optional[int] = None,
    ) -> Layer:
        """
        Given a dataset, this method will save the
        volume annotation of this annotation into that dataset
        by creating a new layer.
        The largest_segment_id is computed automatically, unless provided
        explicitly.

        `volume_annotation_name` has to be provided, if the annotation contains
        multiple volume layers. Use `get_volume_annotation_names()` to look up
        available layers.
        """
        volume_zip_path = self._get_volume_annotation(
            volume_annotation_name=volume_annotation_name,
            volume_annotation_id=volume_annotation_id,
        ).zip

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
                best_mag_view.read(absolute_bounding_box=bbox).max()
                for bbox in best_mag_view.get_bounding_boxes_on_disk()
            )
            layer.largest_segment_id = int(max_value)
        else:
            layer.largest_segment_id = largest_segment_id
        return layer

    @contextmanager
    def temporary_volume_annotation_layer_copy(
        self, volume_annotation_name: Optional[str] = None
    ) -> Iterator[Layer]:

        """
        Given a volume annotation path, create a temporary dataset which
        contains the volume annotation via a symlink. Yield the layer
        so that one can work with the annotation as a wk.Dataset.

        If the annotation contains multiple volume layers, the name of the
        desired volume layer has to be passed via `volume_annotation_name`.
        """

        with TemporaryDirectory() as tmp_annotation_dir:
            tmp_annotation_dataset_path = (
                Path(tmp_annotation_dir) / "tmp_annotation_dataset"
            )

        input_annotation_dataset = Dataset(
            str(tmp_annotation_dataset_path), scale=(1, 1, 1), exist_ok=True
        )

        input_annotation_layer = self.save_volume_annotation(
            input_annotation_dataset,
            "volume_annotation",
            volume_annotation_name=volume_annotation_name,
        )

        yield input_annotation_layer


def open_annotation(annotation_path: Union[str, PathLike]) -> "Annotation":
    if Path(annotation_path).exists():
        warnings.warn(
            "[DEPRECATION] open_annotation is deprecated, please use Annotation.load instead."
        )
        return Annotation.load(annotation_path)
    else:
        assert isinstance(
            annotation_path, str
        ), f"Called open_annotation with a path-like, but {annotation_path} does not exist."
        warnings.warn(
            "[DEPRECATION] open_annotation is deprecated, please use Annotation.download instead."
        )
        return Annotation.download(annotation_path)
