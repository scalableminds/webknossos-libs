import re
import warnings
from contextlib import contextmanager, nullcontext
from enum import Enum, unique
from functools import lru_cache
from io import BytesIO
from os import PathLike
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import (
    IO,
    BinaryIO,
    ContextManager,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    cast,
)
from zipfile import ZipFile

from attr import dataclass
from boltons.cacheutils import cachedproperty

import webknossos.skeleton.nml as wknml
from webknossos.dataset import Dataset, Layer, SegmentationLayer
from webknossos.skeleton import Skeleton

MAG_RE = r"((\d+-\d+-)?\d+)"
SEP_RE = r"(\/|\\)"
CUBE_RE = fr"z\d+{SEP_RE}y\d+{SEP_RE}x\d+\.wkw"
ANNOTATION_WKW_PATH_RE = re.compile(fr"{MAG_RE}{SEP_RE}(header\.wkw|{CUBE_RE})")


class _ZipPath(NamedTuple):
    """Small wrapper around a zipfile.Zipfile object, pointing to a path within this zipfile."""

    zipfile: ZipFile
    path: str  # path within the zipfile itself

    def open(
        self, mode: str = "r", *, pwd: Optional[bytes] = None, force_zip64: bool = True
    ) -> IO[bytes]:
        assert "b" in mode, "Opening a ZipFile currently only supports binary mode"
        zip_mode = mode[0]
        return self.zipfile.open(
            self.path, mode=zip_mode, pwd=pwd, force_zip64=force_zip64
        )


@dataclass
class Annotation:
    file: Union[str, PathLike, BytesIO]

    @cachedproperty
    def _zipfile(self) -> ZipFile:
        return ZipFile(self.file)

    @cachedproperty
    def _filelist(self) -> List[str]:
        return [i.filename for i in self._zipfile.filelist]

    @cachedproperty
    def _nml_file(self) -> _ZipPath:
        nml_files = [i for i in self._filelist if i.endswith(".nml")]
        assert len(nml_files) == 1
        return _ZipPath(self._zipfile, nml_files[0])

    @cachedproperty
    def skeleton(self) -> Skeleton:
        return Skeleton.load(self._nml_file)

    @cachedproperty
    def _nml(self) -> wknml.NML:
        with self._nml_file.open(mode="rb") as file_handle:
            return wknml.parse_nml(file_handle)

    @cachedproperty
    def dataset_name(self) -> str:
        return self.skeleton.name

    @cachedproperty
    def name(self) -> str:
        return self._nml_file.path[:-4]

    @lru_cache(maxsize=128)
    def binary(self) -> Union[bytes, BinaryIO]:
        if isinstance(self.file, BytesIO):
            return self.file.getvalue()
        else:
            return open(self.file, "rb")

    def get_volume_layer_names(self) -> Iterable[str]:

        return (volume.name or str(volume.id) for volume in self._nml.volumes)

    def save_volume_annotation(
        self,
        dataset: Dataset,
        layer_name: str = "volume_annotation",
        largest_segment_id: Optional[int] = None,
        source_volume_name: Optional[str] = None,
    ) -> Layer:
        """
        Given a dataset, this method will save the
        volume annotation of this annotation into that dataset
        by creating a new layer.
        The largest_segment_id is computed automatically, unless provided
        explicitly.

        `source_volume_name` has to be provided, if the annotation contains
        multiple volume layers. Use `get_volume_layer_names()` to look up
        available layers.
        """

        # todo pylint: disable=fixme
        assert len(self._nml.volumes) > 0

        volume_zip_path: Optional[str] = None
        if len(self._nml.volumes) == 1:
            volume_zip_path = self._nml.volumes[0].location
        else:
            assert source_volume_name != None, (
                "The annotation contains multiple volume layers. "
                "Please specify which layer should be used via `source_volume_name`."
            )

            volume_zip_path = None
            for volume in self._nml.volumes:
                if (volume.name or volume.id) == source_volume_name:
                    volume_zip_path = volume.location
                    break
            assert (
                volume_zip_path is not None
            ), f"The specified volume name {source_volume_name} could not be found in this annotation."

        assert (
            volume_zip_path in self._filelist
        ), f"Cannot find {volume_zip_path} in {self._filelist}"
        with self._zipfile.open(volume_zip_path) as f:
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
        min_mag_view = layer.mags[min(layer.mags)]

        if largest_segment_id is None:
            max_value = max(
                min_mag_view.read(absolute_bounding_box=bbox).max()
                for bbox in min_mag_view.get_bounding_boxes_on_disk()
            )
            layer.largest_segment_id = int(max_value)
        else:
            layer.largest_segment_id = largest_segment_id
        return layer

    @classmethod
    def load(cls, annotation_path: Union[str, PathLike]) -> "Annotation":
        assert Path(
            annotation_path
        ).exists(), f"Annotation path {annotation_path} does not exist."
        return Annotation(annotation_path)

    @classmethod
    def download(cls, annotation_path: str) -> "Annotation":
        from webknossos.client.context import _get_context, webknossos_context

        match = re.match(annotation_url_regex, annotation_path)
        assert (
            match is not None
        ), "Annotation.download() must be called with an annotation url, e.g. https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640"
        webknossos_url, annotation_type_str, annotation_id = match.groups()
        annotation_type = AnnotationType(annotation_type_str)

        from webknossos.client._download_annotation import download_annotation

        if webknossos_url != _get_context().url:
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
            return download_annotation(annotation_type, annotation_id)

    @contextmanager
    def _open_nml(self) -> Iterator[IO[bytes]]:
        """
        This method can be used to open the annotation's inner NML
        file directly.
        """
        with self._nml_file.open(mode="rb") as file_handle:
            yield file_handle

    def save(self, path: Union[Path, str]) -> None:
        """
        Stores the annotation as a zip at the given path.
        """

        assert Path(path).suffix == ".zip", "The target path should be a zip file."

        if isinstance(self.file, str) or isinstance(self.file, PathLike):
            copyfile(self.file, str(path))
        else:
            with open(path, "wb") as f:
                f.write(self.file.getbuffer())

    @contextmanager
    def temporary_volume_annotation_layer_copy(
        self, source_volume_name: Optional[str] = None
    ) -> Iterator[Layer]:

        """
        Given a volume annotation path, create a temporary dataset which
        contains the volume annotation via a symlink. Yield the layer
        so that one can work with the annotation as a wk.Dataset.

        If the annotation contains multiple volume layers, the name of the
        desired volume layer has to be passed via `source_volume_name`.
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
            source_volume_name=source_volume_name,
        )

        yield input_annotation_layer


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


annotation_url_regex = re.compile(
    fr"(https?://.*)/annotations/({'|'.join(i.value for i in AnnotationType.__members__.values())})/([0-9A-Fa-f]*)"
)


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
