import copy
import json
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum, unique
from itertools import product
from os import PathLike
from os.path import relpath
from typing import TYPE_CHECKING, Any, Union, cast

import attr
import numpy as np
from cluster_tools import Executor
from natsort import natsort_keygen
from numpy.typing import DTypeLike
from upath import UPath

from ..client.api_client.models import (
    ApiReserveDatasetUplaodToPathsParameters,
    ApiReserveDatasetUploadToPathsForPreliminaryParameters,
)
from ..geometry import (
    BoundingBox,
    Mag,
    NDBoundingBox,
    Vec3Int,
    Vec3IntLike,
    VecIntLike,
)
from ..geometry.mag import MagLike
from ..geometry.nd_bounding_box import derive_nd_bounding_box_from_shape
from ._utils import pims_images
from .abstract_dataset import DEFAULT_VERSION, AbstractDataset
from .defaults import (
    DEFAULT_BIT_DEPTH,
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
    DEFAULT_SHARD_SHAPE_FROM_IMAGES,
    PROPERTIES_FILE_NAME,
    ZARR_JSON_FILE_NAME,
    ZGROUP_FILE_NAME,
)
from .layer import (
    ArrayException,
    ArrayInfo,
    BaseArray,
    LayerToLink,
    RemoteSegmentationLayer,
    Zarr3Config,
)
from .layer.abstract_layer import (
    _UNALLOWED_LAYER_NAME_CHARS,
    _dtype_per_channel_to_element_class,
    _normalize_dtype_per_channel,
    _normalize_dtype_per_layer,
)
from .layer.layer import _get_shard_shape
from .ome_metadata import write_ome_metadata
from .remote_dataset import RemoteDataset
from .remote_folder import RemoteFolder
from .sampling_modes import SamplingModes
from .transfer_mode import TransferMode

if TYPE_CHECKING:
    import pims


from webknossos.dataset.layer import (
    Layer,
    RemoteLayer,
    SegmentationLayer,
)
from webknossos.dataset.layer.abstract_layer import (
    _dtype_per_layer_to_dtype_per_channel,
)

from ..dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    AttachmentsProperties,
    DataFormat,
    DatasetProperties,
    LayerCategoryType,
    LayerProperties,
    SegmentationLayerProperties,
    VoxelSize,
)
from ..dataset_properties.structuring import (
    _properties_floating_type_to_python_type,
    get_dataset_converter,
)
from ..utils import (
    cheap_resolve,
    copytree,
    count_defined_values,
    dump_path,
    enrich_path,
    get_executor_for_args,
    is_fs_path,
    named_partial,
    rmtree,
    strip_trailing_slash,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._utils.infer_bounding_box_existing_files import infer_bounding_box_existing_files
from ._utils.segmentation_recognition import (
    guess_category_from_view,
    guess_if_segmentation_path,
)

logger = logging.getLogger(__name__)

_ALLOWED_COLOR_LAYER_DTYPES = (
    "uint8",
    "uint16",
    "uint32",
    "int8",
    "int16",
    "int32",
    "float32",
)
_ALLOWED_SEGMENTATION_LAYER_DTYPES = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
)

SAFE_LARGE_XY: int = 10_000_000_000  # 10 billion


def _find_array_info(layer_path: UPath) -> ArrayInfo | None:
    for f in layer_path.iterdir():
        if f.is_dir():
            try:
                array = BaseArray.open(f)
                return array.info
            except ArrayException:
                pass
    return None


def _validate_layer_name(layer_name: str) -> None:
    from webknossos.dataset.layer.abstract_layer import _ALLOWED_LAYER_NAME_REGEX

    if _ALLOWED_LAYER_NAME_REGEX.match(layer_name) is None:
        raise ValueError(
            f"The layer name '{layer_name}' is invalid. It must only contain letters, numbers, underscores, hyphens and dots."
        )


class Dataset(AbstractDataset[Layer, SegmentationLayer]):
    """A dataset is the entry point of the Dataset API.

    An existing dataset on disk can be opened or new datasets can be created.

    A dataset stores the data in `.wkw` files on disk with metadata in `datasource-properties.json`.
    The information in those files are kept in sync with the object.

    Each dataset consists of one or more layers (webknossos.dataset.layer.Layer),
    which themselves can comprise multiple magnifications (webknossos.dataset.mag_view.MagView).


    Examples:
        Create a new dataset:
            ```
            ds = Dataset("path/to/dataset", voxel_size=(11.2, 11.2, 25))
            ```

        Open an existing dataset:
            ```
            ds = Dataset.open("path/to/dataset")
            ```

        Open a remote dataset:
            ```
            ds = RemoteDataset.open("my_dataset", "organization_id")
            ```
    """

    @unique
    class ConversionLayerMapping(Enum):
        """Strategies for mapping file paths to layers when importing images.

        These strategies determine how input image files are grouped into layers during
        dataset creation using `Dataset.from_images()`. If no strategy is provided,
        `INSPECT_SINGLE_FILE` is used as the default.

        If none of the pre-defined strategies fit your needs, you can provide a custom
        callable that takes a Path and returns a layer name string.

        Examples:
            Using default strategy:
                ```
                ds = Dataset.from_images("images/", "dataset/")
                ```

            Explicit strategy:
                ```
                ds = Dataset.from_images(
                    "images/",
                    "dataset/",
                    map_filepath_to_layer_name=ConversionLayerMapping.ENFORCE_SINGLE_LAYER
                )
                ```

            Custom mapping function:
                ```
                ds = Dataset.from_images(
                    "images/",
                    "dataset/",
                    map_filepath_to_layer_name=lambda p: p.stem
                )
                ```
        """

        INSPECT_SINGLE_FILE = "inspect_single_file"
        """Default strategy. Inspects first image file to determine if data is 2D or 3D.
        For 2D data uses ENFORCE_LAYER_PER_FOLDER, for 3D uses ENFORCE_LAYER_PER_FILE."""

        INSPECT_EVERY_FILE = "inspect_every_file"
        """Like INSPECT_SINGLE_FILE but determines strategy separately for each file.
        More flexible but slower for many files."""

        ENFORCE_LAYER_PER_FILE = "enforce_layer_per_file"
        """Creates a new layer for each input file. Useful for converting multiple
        3D images or when each 2D image should become its own layer."""

        ENFORCE_SINGLE_LAYER = "enforce_single_layer"
        """Combines all input files into a single layer. Only useful when all
        images are 2D slices that should be combined."""

        ENFORCE_LAYER_PER_FOLDER = "enforce_layer_per_folder"
        """Groups files by their containing folder. Each folder becomes one layer.
        Useful for organized 2D image stacks."""

        ENFORCE_LAYER_PER_TOPLEVEL_FOLDER = "enforce_layer_per_toplevel_folder"
        """Groups files by their top-level folder. Useful when multiple layers each
        have their stacks split across subfolders."""

        def _to_callable(
            self,
            input_path: UPath,
            input_files: Sequence[UPath],
            use_bioformats: bool | None,
        ) -> Callable[[UPath], str]:
            ConversionLayerMapping = Dataset.ConversionLayerMapping

            if self == ConversionLayerMapping.ENFORCE_LAYER_PER_FILE:
                return lambda p: p.as_posix().replace("/", "_")
            elif self == ConversionLayerMapping.ENFORCE_SINGLE_LAYER:
                return lambda _p: input_path.name
            elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_FOLDER:
                return lambda p: (
                    input_path.name
                    if p.parent == UPath()
                    else p.parent.as_posix().replace("/", "_")
                )
            elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_TOPLEVEL_FOLDER:
                return lambda p: input_path.name if p.parent == UPath() else p.parts[0]
            elif self == ConversionLayerMapping.INSPECT_EVERY_FILE:
                # If a file has z dimensions, it becomes its own layer,
                # if it's 2D, the folder becomes a layer.
                return lambda p: (
                    str(p)
                    if pims_images.has_image_z_dimension(
                        input_path / p,
                        use_bioformats=use_bioformats,
                        is_segmentation=guess_if_segmentation_path(p),
                    )
                    else (
                        input_path.name
                        if p.parent == UPath()
                        else p.parent.as_posix().replace("/", "_")
                    )
                )
            elif self == ConversionLayerMapping.INSPECT_SINGLE_FILE:
                # As before, but only a single image is inspected to determine 2D vs 3D.
                if pims_images.has_image_z_dimension(
                    input_path / input_files[0],
                    use_bioformats=use_bioformats,
                    is_segmentation=guess_if_segmentation_path(input_files[0]),
                ):
                    return str
                else:
                    return lambda p: (
                        input_path.name if p.parent == UPath() else p.parts[-2]
                    )
            else:
                raise ValueError(f"Got unexpected ConversionLayerMapping value: {self}")

    def __init__(
        self,
        dataset_path: str | PathLike | UPath,
        voxel_size: tuple[float, float, float] | None = None,  # in nanometers
        name: str | None = None,
        exist_ok: bool = False,
        *,
        voxel_size_with_unit: VoxelSize | None = None,
        read_only: bool = False,
    ) -> None:
        """Create a new dataset or open an existing one.

        Creates a new dataset and the associated `datasource-properties.json` if one does not exist.
        If the dataset already exists and exist_ok is True, it is opened (the provided voxel_size
        and name are asserted to match the existing dataset).

        Please use `Dataset.open` if you intend to open an existing dataset and don't want/need
        the creation behavior.

        Args:
            dataset_path: Path where the dataset should be created/opened
            voxel_size: Optional tuple of floats (x, y, z) specifying voxel size in nanometers
            name: Optional name for the dataset, defaults to last part of dataset_path if not provided
            exist_ok: Whether to open an existing dataset at the path rather than failing
            voxel_size_with_unit: Optional voxel size with unit specification
            read_only: Whether to open dataset in read-only mode

        Raises:
            RuntimeError: If dataset exists and exist_ok=False
            AssertionError: If opening existing dataset with mismatched voxel size or name

        """
        path = strip_trailing_slash(UPath(dataset_path))

        self.path: UPath = path
        self._resolved_path: UPath = cheap_resolve(path)

        if count_defined_values((voxel_size, voxel_size_with_unit)) > 1:
            raise ValueError(
                "Please supply exactly one of voxel_size or voxel_size_with_unit."
            )
        elif voxel_size is not None:
            voxel_size_with_unit = VoxelSize(voxel_size)

        stored_dataset_properties: DatasetProperties | None = None
        try:
            stored_dataset_properties = self._load_dataset_properties_from_path(
                self.path
            )
        except FileNotFoundError:
            if read_only:
                raise FileNotFoundError(
                    f"Cannot open read-only dataset, could not find data at {self.path}."
                )

        dataset_existed_already = stored_dataset_properties is not None
        if dataset_existed_already:
            if not exist_ok:
                raise RuntimeError(
                    f"Creation of Dataset at {self.path} failed, because a non-empty folder already exists at this path."
                )
            assert (
                stored_dataset_properties is not None
            )  # for mypy to get the type of dataset_properties right
            dataset_properties = stored_dataset_properties

        else:
            assert not read_only

            if self.path.exists():
                if self.path.is_dir():
                    if next(self.path.iterdir(), None) is not None:
                        raise RuntimeError(
                            f"Creation of Dataset at {self.path} failed, because a non-empty folder already exists at this path."
                        )
                else:
                    raise NotADirectoryError(
                        f"Creation of Dataset at {self.path} failed, because the given path already exists but is not a directory."
                    )
            # Create directories on disk and write datasource-properties.json
            try:
                self.path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise type(e)(f"Creation of Dataset {self.path} failed. " + repr(e))

            if voxel_size_with_unit is None:
                raise ValueError(
                    "When creating a new dataset, voxel_size or voxel_size_with_unit must be set, e.g. Dataset(path, voxel_size=(1, 1, 4.2))."
                )
            name = name or self.path.absolute().name
            dataset_properties = DatasetProperties(
                id={"name": name, "team": ""},
                scale=voxel_size_with_unit,
                data_layers=[],
                version=DEFAULT_VERSION,
            )

        super().__init__(
            dataset_properties,
            read_only,
        )

        # check if specified parameters match the existing dataset
        if dataset_existed_already:
            if voxel_size_with_unit is None:
                raise ValueError(
                    "Please always supply the voxel_size or voxel_size_with_unit when using the constructor Dataset(your_path, voxel_size=your_voxel_size)."
                    + "If you just want to open an existing dataset, please use Dataset.open(your_path).",
                )
            else:
                if self.voxel_size_with_unit != voxel_size_with_unit:
                    raise RuntimeError(
                        f"Cannot open Dataset: The dataset {self.path} already exists, but the voxel_sizes do not match ({self.voxel_size_with_unit} != {voxel_size_with_unit})"
                    )
            if name is not None:
                if self.name != name:
                    raise RuntimeError(
                        f"Cannot open Dataset: The dataset {self.path} already exists, but the names do not match ({self.name} != {name})"
                    )
        else:
            self._save_dataset_properties(check_existing_properties=False)

    @property
    def _LayerType(self) -> type[Layer]:
        return Layer

    @property
    def _SegmentationLayerType(self) -> type[SegmentationLayer]:
        return SegmentationLayer

    def _initialize_layer_from_properties(
        self, properties: LayerProperties, read_only: bool
    ) -> Layer:
        # If the numChannels key is not present in the dataset properties, assume it is 1 unless we have uint24.
        if properties.num_channels is None:
            if properties.element_class == "uint24":
                properties.num_channels = 3
            else:
                properties.num_channels = 1
        return super()._initialize_layer_from_properties(properties, read_only)

    @classmethod
    def open(
        cls, dataset_path: str | PathLike | UPath, read_only: bool = False
    ) -> "Dataset":
        """
        To open an existing dataset on disk, simply call `Dataset.open("your_path")`.
        This requires `datasource-properties.json` to exist in this folder. Based on the `datasource-properties.json`,
        a dataset object is constructed. Only layers and magnifications that are listed in the properties are loaded
        (even though there might exist more layers or magnifications on disk).

        The `dataset_path` refers to the top level directory of the dataset (excluding layer or magnification names).
        """

        dataset_path = strip_trailing_slash(UPath(dataset_path))
        dataset_properties = cls._load_dataset_properties_from_path(dataset_path)

        dataset = cls.__new__(cls)
        dataset.path = dataset_path
        dataset._resolved_path = cheap_resolve(dataset_path)
        dataset._init_from_properties(dataset_properties, read_only)
        return dataset

    @classmethod
    def open_remote(
        cls,
        dataset_name_or_url: str | None = None,
        organization_id: str | None = None,
        sharing_token: str | None = None,
        webknossos_url: str | None = None,
        dataset_id: str | None = None,
        annotation_id: str | None = None,
        use_zarr_streaming: bool = True,
        read_only: bool = False,
    ) -> "RemoteDataset":
        warn_deprecated("Dataset.open_remote", "RemoteDataset.open")
        return RemoteDataset.open(
            dataset_name_or_url,
            organization_id,
            sharing_token,
            webknossos_url,
            dataset_id,
            annotation_id,
            use_zarr_streaming,
            read_only,
        )

    def __repr__(self) -> str:
        return f"Dataset({repr(self.path)})"

    def _load_dataset_properties(self) -> DatasetProperties:
        """
        Loads the current dataset properties from json on disk.
        """
        return self._load_dataset_properties_from_path(self.path)

    def _save_dataset_properties_impl(
        self, layer_renaming: tuple[str, str] | None = None
    ) -> None:
        """
        Exports the current dataset properties to json on disk.
        And writes out Zarr and OME-Ngff metadata if there is a Zarr layer.
        """
        del layer_renaming  # only used in remote case
        (self.path / PROPERTIES_FILE_NAME).write_text(
            json.dumps(
                get_dataset_converter().unstructure(self._properties),
                indent=4,
            ),
        )

        # Write out Zarr and OME-Ngff metadata if there is a Zarr layer
        if any(layer.data_format == DataFormat.Zarr for layer in self.layers.values()):
            (self.path / ZGROUP_FILE_NAME).write_text(
                json.dumps({"zarr_format": "2"}, indent=4),
            )
        if any(layer.data_format == DataFormat.Zarr3 for layer in self.layers.values()):
            (self.path / ZARR_JSON_FILE_NAME).write_text(
                json.dumps({"zarr_format": 3, "node_type": "group"}, indent=4),
            )

        for layer in self.layers.values():
            # Only write out OME metadata if the layer is a child of the dataset
            if not layer.is_foreign and layer.path.exists():
                write_ome_metadata(self, layer)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.path == other.path and self.read_only == other.read_only
        else:
            return False

    @property
    def resolved_path(self) -> UPath:
        return self._resolved_path

    @classmethod
    def download(
        cls,
        dataset_name_or_url: str,
        *,
        organization_id: str | None = None,
        sharing_token: str | None = None,
        webknossos_url: str | None = None,
        bbox: BoundingBox | None = None,
        layers: list[str] | str | None = None,
        mags: list[Mag] | None = None,
        path: PathLike | UPath | str | None = None,
        exist_ok: bool = False,
    ) -> "Dataset":
        """Downloads a dataset and returns the Dataset instance.

        * `dataset_name_or_url` may be a dataset name or a full URL to a dataset view, e.g.
          `https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view`
          If a URL is used, `organization_id`, `webknossos_url` and `sharing_token` must not be set.
        * `organization_id` may be supplied if a dataset name was used in the previous argument,
          it defaults to your current organization from the `webknossos_context`.
          You can find your `organization_id` [here](https://webknossos.org/auth/token).
        * `sharing_token` may be supplied if a dataset name was used and can specify a sharing token.
        * `webknossos_url` may be supplied if a dataset name was used,
          and allows to specify in which webknossos instance to search for the dataset.
          It defaults to the url from your current `webknossos_context`, using https://webknossos.org as a fallback.
        * `bbox`, `layers`, and `mags` specify which parts of the dataset to download.
          If nothing is specified the whole image, all layers, and all mags are downloaded respectively.
        * `path` and `exist_ok` specify where to save the downloaded dataset and whether to overwrite
          if the `path` exists.
        """

        warn_deprecated("Dataset.download", "RemoteDataset.download")

        remote_dataset = RemoteDataset.open(
            dataset_name_or_url, organization_id, sharing_token, webknossos_url
        )
        return remote_dataset.download(
            sharing_token, bbox, layers, mags, path, exist_ok
        )

    def publish_to_preliminary_dataset(
        self,
        dataset_id: str,
        path_prefix: str | None = None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> None:
        """
        Copies or moves+symlinks the data to paths returned by WEBKNOSSOS
        The dataset needs to be in status "uploading".
        The dataset already exists in WEBKNOSSOS but has no dataset_properties.
        With the dataset_properties WEBKNOSSOS can reserve the paths.
        Args:
            dataset_id: The dataset_id of the already existing dataset
            path_prefix: The prefix of the storage path, can be used to select one of the storage path options.
            symlink_data_instead_of_copy: Set to true if the client has access to the same file system as the WEBKNOSSOS datastore.
        """
        from ..client.context import _get_api_client

        if transfer_mode == TransferMode.HTTP:
            raise ValueError("HTTP transfer mode is not supported for this method")

        client = _get_api_client()
        response = client.reserve_dataset_upload_to_paths_for_preliminary(
            dataset_id,
            ApiReserveDatasetUploadToPathsForPreliminaryParameters(
                self._properties, path_prefix
            ),
        )
        self._transfer_dataset_items(response.data_source, transfer_mode)

        client.finish_dataset_upload_to_paths(dataset_id)

    def upload(
        self,
        new_dataset_name: str | None = None,
        initial_team_ids: list[str] | None = None,
        folder_id: str | RemoteFolder | None = None,
        require_unique_name: bool = False,
        layers_to_link: Sequence[LayerToLink | RemoteLayer] | None = None,
        transfer_mode: TransferMode = TransferMode.HTTP,
        jobs: int | None = None,
        common_storage_path_prefix: str | None = None,
    ) -> "RemoteDataset":
        """Upload this dataset to webknossos.

        Creates database entries and sets access rights on the webknossos instance before the actual data upload.
        The client then copies the data directly to the returned paths.

        Args:
            new_dataset_name: Name for the new dataset defaults to the current name.
            initial_team_ids: Optional list of team IDs to grant initial access
            folder_id: Optional ID of folder where dataset should be placed
            require_unique_name: Whether to make request fail in case a dataset with the name already exists
            layers_to_link: Optional list of LayerToLink to link already published layers to the dataset.
            jobs: Optional number of jobs to use for uploading the data.
            common_storage_path_prefix: Optional path prefix used when transfer_mode is either COPY or MOVE_AND_SYMLINK
                                        to select one of the available WEBKNOSSOS storages.
        Returns:
            RemoteDataset: Reference to the newly created remote dataset
        Note:
            upload_directly_to_common_storage is typically only used by administrators with direct file system or S3 access to the WEBKNOSSOS datastore.
            Most users should let upload_directly_to_common_storage to default to False
        Examples:
            ```
            remote_ds = ds.upload(
                "my_dataset",
                ["team_a", "team_b"],
                "folder_123"
            )
            print(remote_ds.url)
            ```
            Link existing layers:
            ```
            link = LayerToLink.from_remote_layer(existing_layer)
            remote_ds = ds.upload(layers_to_link=[link])
            ```
        """

        from ..client.context import _get_api_client

        new_dataset_name = new_dataset_name or self.name

        if isinstance(folder_id, RemoteFolder):
            folder_id = folder_id.id

        converted_layers_to_link = (
            None
            if layers_to_link is None
            else [
                i if isinstance(i, LayerToLink) else LayerToLink.from_remote_layer(i)
                for i in layers_to_link
            ]
        )

        if transfer_mode in (
            TransferMode.COPY,
            TransferMode.MOVE_AND_SYMLINK,
            TransferMode.SYMLINK,
        ):
            client = _get_api_client()
            response = client.reserve_dataset_upload_to_paths(
                ApiReserveDatasetUplaodToPathsParameters(
                    dataset_name=new_dataset_name,
                    initial_team_ids=initial_team_ids or [],
                    folder_id=folder_id,
                    require_unique_name=require_unique_name,
                    data_source=self._properties,
                    layers_to_link=[
                        layer_to_link._as_api_linked_layer_identifier()
                        for layer_to_link in converted_layers_to_link
                    ]
                    if converted_layers_to_link
                    else [],
                    path_prefix=common_storage_path_prefix,
                )
            )
            new_dataset_id = response.new_dataset_id
            data_source = response.data_source

            self._transfer_dataset_items(data_source, transfer_mode)
            # announce finished upload
            client.finish_dataset_upload_to_paths(new_dataset_id)
        else:
            assert transfer_mode == TransferMode.HTTP, "Expected HTTP transfer mode"
            if common_storage_path_prefix is not None:
                logger.warning(
                    "common_storage_path_prefix is not None, but uploading via HTTP."
                    " Ignoring common_storage_path_prefix."
                )
            for layer in self.get_segmentation_layers():
                if not layer.attachments.is_empty:
                    raise NotImplementedError(
                        f"Uploading layers with attachments is not supported yet. Layer {layer.name} has attachments."
                    )

            from ..client._upload_dataset import upload_dataset

            new_dataset_id = upload_dataset(
                self,
                new_dataset_name,
                converted_layers_to_link,
                jobs,
                folder_id=folder_id,
            )

        return RemoteDataset.open(dataset_id=new_dataset_id)

    def _transfer_dataset_items(
        self, data_source: DatasetProperties, transfer_mode: TransferMode
    ) -> None:
        """
        Iterates over the mags and attachments and transfers them to the target location.
        """
        assert transfer_mode in (
            TransferMode.COPY,
            TransferMode.MOVE_AND_SYMLINK,
            TransferMode.SYMLINK,
        ), f"transfer mode not supported. found {transfer_mode}"

        # transfer data
        for layer in data_source.data_layers:
            src_layer = self.layers[layer.name]
            for mag in layer.mags:
                src_mag = src_layer.mags[mag.mag]
                assert mag.path is not None, "mag.path must be set to copy/move data"
                dst_mag_path = enrich_path(mag.path)
                transfer_mode.transfer(
                    src_mag.path, dst_mag_path, progress_desc_label="mag"
                )
            if isinstance(src_layer, SegmentationLayer):
                assert isinstance(layer, SegmentationLayerProperties), (
                    "If src_layer is a SegmentationLayer, then layer must be a SegmentationLayerProperties"
                )
                # iterate over attachments
                for src_attachment, dst_attachment in zip(
                    src_layer.attachments, layer.attachments
                ):
                    dst_attachment_path = enrich_path(dst_attachment.path)
                    transfer_mode.transfer(
                        src_attachment.path,
                        dst_attachment_path,
                        progress_desc_label="attachment",
                    )

    @staticmethod
    def get_remote_datasets(
        *,
        organization_id: str | None = None,
        tags: str | Sequence[str] | None = None,
        name: str | None = None,
        folder_id: RemoteFolder | str | None = None,
    ) -> Mapping[str, "RemoteDataset"]:
        warn_deprecated("Dataset.get_remote_datasets", "RemoteDataset.list")
        from webknossos import RemoteDataset

        return RemoteDataset.list(organization_id, tags, name, folder_id)

    @classmethod
    def trigger_reload_in_datastore(
        cls,
        dataset_name_or_url: str | None = None,
        organization_id: str | None = None,
        webknossos_url: str | None = None,
        dataset_id: str | None = None,
        organization: str | None = None,
        datastore_url: str | None = None,
        token: str | None = None,
    ) -> None:
        warn_deprecated(
            "Dataset.trigger_reload_in_datastore",
            "RemoteDataset.trigger_reload_in_datastore",
        )
        RemoteDataset.trigger_reload_in_datastore(
            dataset_name_or_url=dataset_name_or_url,
            organization_id=organization_id,
            webknossos_url=webknossos_url,
            dataset_id=dataset_id,
            organization=organization,
            datastore_url=datastore_url,
            token=token,
        )

    @classmethod
    def trigger_dataset_import(
        cls, directory_name: str, organization: str, token: str
    ) -> None:
        """Deprecated. Use `Dataset.trigger_reload_in_datastore` instead."""
        warn_deprecated(
            "trigger_dataset_import", "RemoteDataset.trigger_reload_in_datastore"
        )

        cls.trigger_reload_in_datastore(
            dataset_name_or_url=directory_name,
            organization_id=organization,
            token=token,
        )

    @classmethod
    def from_images(
        cls,
        input_path: str | PathLike | UPath,
        output_path: str | PathLike | UPath,
        voxel_size: tuple[float, float, float] | None = None,
        name: str | None = None,
        *,
        map_filepath_to_layer_name: ConversionLayerMapping
        | Callable[[UPath], str] = ConversionLayerMapping.INSPECT_SINGLE_FILE,
        z_slices_sort_key: Callable[[UPath], Any] = natsort_keygen(),
        voxel_size_with_unit: VoxelSize | None = None,
        layer_name: str | None = None,
        layer_category: LayerCategoryType | None = None,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: int | Vec3IntLike | None = None,
        compress: bool = True,
        swap_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        use_bioformats: bool | None = None,
        max_layers: int = 20,
        batch_size: int | None = None,
        executor: Executor | None = None,
    ) -> "Dataset":
        """This method imports image data in a folder or from a file as a webknossos dataset.

        The image data can be 3D images (such as multipage tiffs) or stacks of 2D images.
        Multiple 3D images or image stacks are mapped to different layers based on the mapping strategy.

        The exact mapping is handled by the argument `map_filepath_to_layer_name`, which can be a pre-defined
        strategy from the enum `ConversionLayerMapping`, or a custom callable, taking
        a path of an image file and returning the corresponding layer name. All
        files belonging to the same layer name are then grouped. In case of
        multiple files per layer, those are usually mapped to the z-dimension.
        The order of the z-slices can be customized by setting
        `z_slices_sort_key`.

        For more fine-grained control, please create an empty dataset and use `add_layer_from_images`.

        Args:
            input_path: Path to input image files
            output_path: Output path for created dataset
            voxel_size: Optional tuple of floats (x,y,z) for voxel size in nm
            name: Optional name for dataset
            map_filepath_to_layer_name: Strategy for mapping files to layers, either a ConversionLayerMapping
                enum value or callable taking Path and returning str
            z_slices_sort_key: Optional key function for sorting z-slices
            voxel_size_with_unit: Optional voxel size with unit specification
            layer_name: Optional name for layer(s)
            layer_category: Optional category override (LayerCategoryType.color / LayerCategoryType.segmentation)
            data_format: Format to store data in ('wkw'/'zarr'/'zarr3)
            chunk_shape: Optional. Shape of chunks to store data in
            shard_shape: Optional. Shape of shards to store data in
            chunks_per_shard: Deprecated, use shard_shape. Optional. number of chunks per shard
            compress: Whether to compress the data
            swap_xy: Whether to swap x and y axes
            flip_x: Whether to flip the x axis
            flip_y: Whether to flip the y axis
            flip_z: Whether to flip the z axis
            use_bioformats: Whether to use bioformats for reading
            max_layers: Maximum number of layers to create
            batch_size: Size of batches for processing
            executor: Optional executor for parallelization

        Returns:
            Dataset: The created dataset instance

        Examples:
            ```
            ds = Dataset.from_images("path/to/images/",
                                    "path/to/dataset/",
                                    voxel_size=(1, 1, 1))
            ```

        Note:
            This method needs extra packages like tifffile or pylibczirw.
            Install with `pip install "webknossos[all]"` and `pip install --extra-index-url https://pypi.scm.io/simple/ "webknossos[czi]"`.
        """

        input_upath = UPath(input_path)

        valid_suffixes = pims_images.get_valid_pims_suffixes()
        if use_bioformats is not False:
            valid_suffixes.update(pims_images.get_valid_bioformats_suffixes())

        if input_upath.is_file():
            if input_upath.suffix.lstrip(".").lower() in valid_suffixes:
                input_files = [UPath(input_upath.name)]
                input_upath = input_upath.parent
        else:
            input_files = [
                i.relative_to(input_upath)
                for i in input_upath.glob("**/*")
                if i.is_file() and i.suffix.lstrip(".").lower() in valid_suffixes
            ]

        if len(input_files) == 0:
            raise ValueError(
                "Could not find any supported image data. "
                + f"The following suffixes are supported: {sorted(valid_suffixes)}"
            )

        if isinstance(map_filepath_to_layer_name, Dataset.ConversionLayerMapping):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="pims",
                )
                warnings.filterwarnings(
                    "once",
                    category=UserWarning,
                    module="pims_images",
                )
                map_filepath_to_layer_name_func = (
                    map_filepath_to_layer_name._to_callable(
                        input_upath,
                        input_files=input_files,
                        use_bioformats=use_bioformats,
                    )
                )
        else:
            map_filepath_to_layer_name_func = map_filepath_to_layer_name
        if voxel_size_with_unit is None:
            assert voxel_size is not None, (
                "Please supply either voxel_size or voxel_size_with_unit."
            )
            voxel_size_with_unit = VoxelSize(voxel_size)
        else:
            assert voxel_size is None, (
                "Please supply either voxel_size or voxel_size_with_unit not both."
            )

        ds = cls(output_path, voxel_size_with_unit=voxel_size_with_unit, name=name)

        filepaths_per_layer: dict[str, list[UPath]] = {}
        for input_file in input_files:
            layer_name_from_mapping = map_filepath_to_layer_name_func(input_file)
            # Remove characters from layer name that are not allowed
            layer_name_from_mapping = _UNALLOWED_LAYER_NAME_CHARS.sub(
                "", layer_name_from_mapping
            )
            # Ensure layer name does not start with a dot
            layer_name_from_mapping = layer_name_from_mapping.lstrip(".")

            assert layer_name_from_mapping != "", (
                f"Could not determine a layer name for {input_file}."
            )

            filepaths_per_layer.setdefault(layer_name_from_mapping, []).append(
                input_upath / input_file
            )

        if layer_name is not None:
            if len(filepaths_per_layer) == 1:
                filepaths_per_layer[layer_name] = filepaths_per_layer.pop(
                    layer_name_from_mapping
                )
            else:
                filepaths_per_layer = {
                    f"{layer_name}_{k}": v for k, v in filepaths_per_layer.items()
                }
        with get_executor_for_args(None, executor) as executor:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="pims_images",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="pims",
                )
                for layer_name, filepaths in filepaths_per_layer.items():
                    filepaths.sort(key=z_slices_sort_key)

                    ds.add_layer_from_images(
                        filepaths[0] if len(filepaths) == 1 else filepaths,
                        layer_name,
                        category=layer_category,
                        data_format=data_format,
                        chunk_shape=chunk_shape,
                        shard_shape=shard_shape,
                        chunks_per_shard=chunks_per_shard,
                        compress=compress,
                        swap_xy=swap_xy,
                        flip_x=flip_x,
                        flip_y=flip_y,
                        flip_z=flip_z,
                        use_bioformats=use_bioformats,
                        batch_size=batch_size,
                        allow_multiple_layers=True,
                        max_layers=max_layers - len(ds.layers),
                        truncate_rgba_to_rgb=False,
                        executor=executor,
                    )

        return ds

    def add_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        *,
        dtype_per_layer: DTypeLike | None = None,
        dtype_per_channel: DTypeLike | None = None,
        num_channels: int | None = None,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        bounding_box: NDBoundingBox | None = None,
        **kwargs: Any,
    ) -> Layer:
        """Create a new layer in the dataset.

        Creates a new layer with the given name, category, and data type.

        Args:
            layer_name: Name for the new layer
            category: Either 'color' or 'segmentation'
            dtype_per_layer: Deprecated, use dtype_per_channel. Optional data type for entire layer, e.g. np.uint8
            dtype_per_channel: Optional data type per channel, e.g. np.uint8
            num_channels: Number of channels (default 1)
            data_format: Format to store data ('wkw', 'zarr', 'zarr3')
            bounding_box: Optional initial bounding box of layer
            **kwargs: Additional arguments:
                - largest_segment_id: For segmentation layers, initial largest ID
                - mappings: For segmentation layers, optional ID mappings

        Returns:
            Layer: The newly created layer

        Raises:
            IndexError: If layer with given name already exists
            RuntimeError: If invalid category specified
            AttributeError: If both dtype_per_layer and dtype_per_channel specified
            AssertionError: If invalid layer name or WKW format used with remote dataset

        Examples:
            Create color layer:
                ```
                layer = ds.add_layer(
                    "my_raw_microscopy_layer",
                    LayerCategoryType.COLOR_CATEGORY,
                    dtype_per_channel=np.uint8,
                )
                ```

            Create segmentation layer:
                ```
                layer = ds.add_layer(
                    "my_segmentation_labels",
                    LayerCategoryType.SEGMENTATION_CATEGORY,
                    dtype_per_channel=np.uint64
                )
                ```

        Note:
            The dtype can be specified either per layer or per channel, but not both.
            If neither is specified, uint8 per channel is used by default.
            WKW format can only be used with local datasets.
        """

        self._ensure_writable()

        _validate_layer_name(layer_name)

        if num_channels is None:
            num_channels = 1

        if dtype_per_layer is not None and dtype_per_channel is not None:
            raise AttributeError(
                "Cannot add layer. Specifying both 'dtype_per_layer' and 'dtype_per_channel' is not allowed"
            )
        elif dtype_per_channel is not None:
            dtype_per_channel = _properties_floating_type_to_python_type.get(
                dtype_per_channel,  # type: ignore[arg-type]
                dtype_per_channel,  # type: ignore[arg-type]
            )
            dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)  # type: ignore[arg-type]
        elif dtype_per_layer is not None:
            warn_deprecated("dtype_per_layer", "dtype_per_channel")
            dtype_per_layer = _properties_floating_type_to_python_type.get(
                dtype_per_layer,  # type: ignore[arg-type]
                dtype_per_layer,  # type: ignore[arg-type]
            )
            dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)  # type: ignore[arg-type]
            dtype_per_channel = _dtype_per_layer_to_dtype_per_channel(
                dtype_per_layer, num_channels
            )
        else:
            dtype_per_channel = np.dtype("uint" + str(DEFAULT_BIT_DEPTH))

        # assert that the dtype_per_channel is supported by webknossos
        if category == COLOR_CATEGORY:
            if dtype_per_channel.name not in _ALLOWED_COLOR_LAYER_DTYPES:
                raise ValueError(
                    f"Cannot add color layer with dtype {dtype_per_channel.name}. "
                    f"Supported dtypes are: {', '.join(_ALLOWED_COLOR_LAYER_DTYPES)}."
                    "For an overview of supported dtypes, see https://docs.webknossos.org/webknossos/data/upload_ui.html",
                )
        else:
            if dtype_per_channel.name not in _ALLOWED_SEGMENTATION_LAYER_DTYPES:
                raise ValueError(
                    f"Cannot add segmentation layer with dtype {dtype_per_channel.name}. "
                    f"Supported dtypes are: {', '.join(_ALLOWED_SEGMENTATION_LAYER_DTYPES)}."
                    "For an overview of supported dtypes, see https://docs.webknossos.org/webknossos/data/upload_ui.html",
                )

        if layer_name in self.layers.keys():
            raise IndexError(
                f"Adding layer {layer_name} failed. There is already a layer with this name"
            )

        assert is_fs_path(self.path) or data_format != DataFormat.WKW, (
            "Cannot create WKW layers in remote datasets. Use `data_format='zarr'`."
        )

        layer_properties = LayerProperties(
            name=layer_name,
            category=category,
            bounding_box=bounding_box or BoundingBox((0, 0, 0), (0, 0, 0)),
            element_class=_dtype_per_channel_to_element_class(
                dtype_per_channel, num_channels
            ),
            mags=[],
            num_channels=num_channels,
            data_format=DataFormat(data_format),
        )

        if category == COLOR_CATEGORY:
            self._properties.data_layers += [layer_properties]
            self._layers[layer_name] = Layer(self, layer_properties, read_only=False)
        elif category == SEGMENTATION_CATEGORY:
            segmentation_layer_properties: SegmentationLayerProperties = (
                SegmentationLayerProperties(
                    **(
                        attr.asdict(layer_properties, recurse=False)
                    ),  # use all attributes from LayerProperties
                    largest_segment_id=kwargs.get("largest_segment_id"),
                )
            )
            if "mappings" in kwargs:
                segmentation_layer_properties.mappings = kwargs["mappings"]
            self._properties.data_layers += [segmentation_layer_properties]
            self._layers[layer_name] = SegmentationLayer(
                self, segmentation_layer_properties, read_only=False
            )
        else:
            raise RuntimeError(
                f"Failed to add layer ({layer_name}) because of invalid category ({category}). The supported categories are '{COLOR_CATEGORY}' and '{SEGMENTATION_CATEGORY}'"
            )

        self._save_dataset_properties()
        return self.layers[layer_name]

    def get_or_add_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        *,
        dtype_per_layer: DTypeLike | None = None,
        dtype_per_channel: DTypeLike | None = None,
        num_channels: int | None = None,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        **kwargs: Any,
    ) -> Layer:
        """Get an existing layer or create a new one.

        Gets a layer with the given name if it exists, otherwise creates a new layer
        with the specified parameters.

        Args:
            layer_name: Name of the layer to get or create
            category: Layer category ('color' or 'segmentation')
            dtype_per_layer: Deprecated, use dtype_per_channel. Optional data type for entire layer
            dtype_per_channel: Optional data type per channel
            num_channels: Optional number of channels
            data_format: Format to store data ('wkw', 'zarr', etc.)
            **kwargs: Additional arguments passed to add_layer()

        Returns:
            Layer: The existing or newly created layer

        Raises:
            AssertionError: If existing layer's properties don't match specified parameters
            ValueError: If both dtype_per_layer and dtype_per_channel specified
            RuntimeError: If invalid category specified

        Examples:
            ```
            layer = ds.get_or_add_layer(
                "segmentation",
                LayerCategoryType.SEGMENTATION_CATEGORY,
                dtype_per_channel=np.uint64,
            )
            ```

        Note:
            The dtype can be specified either per layer or per channel, but not both.
            For existing layers, the parameters are validated against the layer properties.
        """

        if layer_name in self.layers.keys():
            assert (
                num_channels is None
                or self.layers[layer_name].num_channels == num_channels
            ), (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the number of channels do not match. "
                + f"The number of channels of the existing layer are '{self.layers[layer_name].num_channels}' "
                + f"and the passed parameter is '{num_channels}'."
            )
            assert self.get_layer(layer_name).category == category, (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the categories do not match. "
                + f"The category of the existing layer is '{self.get_layer(layer_name).category}' "
                + f"and the passed parameter is '{category}'."
            )

            if dtype_per_channel is not None:
                dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)

            if dtype_per_layer is not None:
                warn_deprecated("dtype_per_layer", "dtype_per_channel")
                dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)

            if dtype_per_channel is not None or dtype_per_layer is not None:
                dtype_per_channel = (
                    dtype_per_channel
                    or _dtype_per_layer_to_dtype_per_channel(
                        dtype_per_layer,  # type: ignore[arg-type]
                        num_channels or self.layers[layer_name].num_channels,
                    )
                )
                assert (
                    dtype_per_channel is None
                    or self.layers[layer_name].dtype_per_channel == dtype_per_channel
                ), (
                    f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the dtypes do not match. "
                    + f"The dtype_per_channel of the existing layer is '{self.layers[layer_name].dtype_per_channel}' "
                    + f"and the passed parameter would result in a dtype_per_channel of '{dtype_per_channel}'."
                )
            return self.layers[layer_name]
        else:
            return self.add_layer(
                layer_name,
                category,
                dtype_per_layer=dtype_per_layer,
                dtype_per_channel=dtype_per_channel,
                num_channels=num_channels,
                data_format=DataFormat(data_format),
                **kwargs,
            )

    def add_layer_like(
        self, other_layer: Layer | RemoteLayer, layer_name: str
    ) -> Layer:
        self._ensure_writable()

        _validate_layer_name(layer_name)

        if layer_name in self.layers.keys():
            raise IndexError(
                f"Adding layer {layer_name} failed. There is already a layer with this name"
            )

        layer_properties = copy.copy(other_layer._properties)
        layer_properties.mags = []
        if isinstance(layer_properties, SegmentationLayerProperties):
            layer_properties.attachments = AttachmentsProperties()
        layer_properties.name = layer_name

        self._properties.data_layers += [layer_properties]
        if layer_properties.category == COLOR_CATEGORY:
            self._layers[layer_name] = Layer(self, layer_properties, read_only=False)
        elif layer_properties.category == SEGMENTATION_CATEGORY:
            self._layers[layer_name] = SegmentationLayer(
                self,
                cast(SegmentationLayerProperties, layer_properties),
                read_only=False,
            )
        else:
            raise RuntimeError(
                f"Failed to add layer ({layer_name}) because of invalid category ({layer_properties.category}). The supported categories are '{COLOR_CATEGORY}' and '{SEGMENTATION_CATEGORY}'"
            )
        self._save_dataset_properties()
        return self._layers[layer_name]

    def _add_existing_layer(self, layer_properties: LayerProperties) -> Layer:
        self._ensure_writable()

        assert layer_properties.name not in self.layers, (
            f"Cannot import layer `{layer_properties.name}` into dataset, "
            + "as a layer with this name is already present."
        )

        self._properties.data_layers.append(layer_properties)
        layer = self._initialize_layer_from_properties(
            layer_properties, read_only=False
        )
        self._layers[layer.name] = layer

        self._save_dataset_properties()
        return self.layers[layer.name]

    def add_layer_for_existing_files(
        self,
        layer_name: str,
        category: LayerCategoryType,
        **kwargs: Any,
    ) -> Layer:
        """Create a new layer from existing data files.

        Adds a layer by discovering and incorporating existing data files that were created externally,
        rather than creating new ones. The layer properties are inferred from the existing files
        unless overridden.

        Args:
            layer_name: Name for the new layer
            category: Layer category ('color' or 'segmentation')
            **kwargs: Additional arguments:
                - num_channels: Override detected number of channels
                - dtype_per_channel: Override detected data type
                - data_format: Override detected data format
                - bounding_box: Override detected bounding box

        Returns:
            Layer: The newly created layer referencing the existing files

        Raises:
            AssertionError: If layer already exists or no valid files found
            RuntimeError: If dataset is read-only

        Examples:
            Basic usage:
                ```
                layer = ds.add_layer_for_existing_files(
                    "external_data",
                    "color"
                )
                ```

            Override properties:
                ```
                layer = ds.add_layer_for_existing_files(
                    "segmentation_data",
                    "segmentation",
                    dtype_per_channel=np.uint64
                )
                ```

        Note:
            The data files must already exist in the dataset directory under the layer name.
            Files are analyzed to determine properties like data type and number of channels.
            Magnifications are discovered automatically.
        """
        self._ensure_writable()

        _validate_layer_name(layer_name)
        assert layer_name not in self.layers, f"Layer {layer_name} already exists!"

        array_info = _find_array_info(self.path / layer_name)
        assert array_info is not None, (
            f"Could not find any valid mags in {self.path / layer_name}. Cannot add layer."
        )

        num_channels = kwargs.pop("num_channels", array_info.num_channels)
        dtype_per_channel = kwargs.pop("dtype_per_channel", array_info.voxel_type)
        data_format = kwargs.pop("data_format", array_info.data_format)

        layer = self.add_layer(
            layer_name,
            category=category,
            num_channels=num_channels,
            dtype_per_channel=dtype_per_channel,
            data_format=data_format,
            **kwargs,
        )

        for mag_dir in layer.path.iterdir():
            try:
                # Tests if directory entry is a valid mag.
                # Metadata files such as zarr.json are filtered out by this.
                Mag(mag_dir.name)
            except ValueError:
                continue
            # Mags are only writable if they are local to the dataset
            resolved_mag_path = cheap_resolve(mag_dir)
            read_only = resolved_mag_path.parent != self.resolved_path / layer_name
            layer._add_mag_for_existing_files(
                mag_dir.name, mag_path=resolved_mag_path, read_only=read_only
            )
        finest_mag_view = layer.mags[min(layer.mags)]
        if "bounding_box" not in kwargs:
            layer.bounding_box = infer_bounding_box_existing_files(finest_mag_view)
        else:
            layer.bounding_box = kwargs["bounding_box"]
        return layer

    def add_layer_from_images(
        self,
        images: Union[str, "pims.FramesSequence", list[str | PathLike | UPath]],
        ## add_layer arguments
        layer_name: str,
        category: LayerCategoryType | None = "color",
        *,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        ## add_mag arguments
        mag: MagLike = Mag(1),
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: int | Vec3IntLike | None = None,
        compress: bool = True,
        ## other arguments
        topleft: VecIntLike = Vec3Int.zeros(),  # in Mag(1)
        swap_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        dtype: DTypeLike | None = None,
        use_bioformats: bool | None = None,
        channel: int | None = None,
        timepoint: int | None = None,
        czi_channel: int | None = None,
        batch_size: int | None = None,  # defaults to shard-size z
        allow_multiple_layers: bool = False,
        max_layers: int = 20,
        truncate_rgba_to_rgb: bool = True,
        executor: Executor | None = None,
    ) -> Layer:
        """
        Creates a new layer called `layer_name` with mag `mag` from `images`.
        `images` can be one of the following:

        * glob-string
        * list of paths
        * `pims.FramesSequence` instance

        Please see the [pims docs](http://soft-matter.github.io/pims/v0.6.1/opening_files.html) for more information.

        This method needs extra packages like tifffile or pylibczirw. Please install the respective extras,
        e.g. using `python -m pip install "webknossos[all]"`.

        Further Arguments:

        * `category`: `color` by default, may be set to "segmentation"
        * `data_format`: by default zarr3 files are written, may be set to "wkw" or "zarr" to write in these formats.
        * `mag`: magnification to use for the written data
        * `chunk_shape`, `chunks_per_shard`, `shard_shape`, `compress`: adjust how the data is stored on disk
        * `topleft`: set an offset in Mag(1) to start writing the data, only affecting the output
        * `swap_xy`: set to `True` to interchange x and y axis before writing to disk
        * `flip_x`, `flip_y`, `flip_z`: set to `True` to reverse the respective axis before writing to disk
        * `dtype`: the read image data will be convertoed to this dtype using `numpy.ndarray.astype`
        * `use_bioformats`: set to `True` to only use the
          [pims bioformats adapter](https://soft-matter.github.io/pims/v0.6.1/bioformats.html) directly, needs a JVM,
          set to `False` to forbid using the bioformats adapter, by default it is tried as a last option
        * `channel`: may be used to select a single channel, if multiple are available
        * `timepoint`: for timeseries, select a timepoint to use by specifying it as an int, starting from 0
        * `czi_channel`: may be used to select a channel for .czi images, which differs from normal color-channels
        * `batch_size`: size to process the images (influences RAM consumption), must be a multiple of the chunk-size z-axis for uncompressed and the shard-size z-axis for compressed layers, default is the chunk-size or shard-size respectively
        * `allow_multiple_layers`: set to `True` if timepoints or channels may result in multiple layers being added (only the first is returned)
        * `max_layers`: only applies if `allow_multiple_layers=True`, limits the number of layers added via different channels or timepoints
        * `truncate_rgba_to_rgb`: only applies if `allow_multiple_layers=True`, set to `False` to write four channels into layers instead of an RGB channel
        * `executor`: pass a `ClusterExecutor` instance to parallelize the conversion jobs across the batches
        """

        _validate_layer_name(layer_name)
        if category is None:
            image_path_for_category_guess: UPath
            if (
                isinstance(images, str)
                or isinstance(images, PathLike)
                or isinstance(images, UPath)
            ):
                image_path_for_category_guess = UPath(images)
            else:
                image_path_for_category_guess = UPath(images[0])
            category = (
                "segmentation"
                if guess_if_segmentation_path(image_path_for_category_guess)
                else "color"
            )
            user_set_category = False
        else:
            user_set_category = True

        pims_image_sequence = pims_images.PimsImages(
            images,
            channel=channel,
            timepoint=timepoint,
            czi_channel=czi_channel,
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            use_bioformats=use_bioformats,
            is_segmentation=category == "segmentation",
        )
        possible_layers = pims_image_sequence.get_possible_layers()
        # Check if 4 color channels should be converted to
        # 3 color channels (rbg)
        if (
            possible_layers is not None
            and truncate_rgba_to_rgb
            and len(possible_layers.get("channel", [])) == 4
        ):
            # Remove channels from possible_layers to keep those
            # and automatically truncate to 3 channels
            # (pims_images takes care of this:)
            del possible_layers["channel"]
        # Further below, we iterate over suffix_with_pims_open_kwargs_per_layer in the for-loop
        # to add one layer per possible_layer if allow_multiple_layers is True.
        # If just a single layer is added, we still add a default value in the dict.
        if possible_layers is not None and len(possible_layers) > 0:
            if allow_multiple_layers:
                # Get all combinations of possible layers. E.g.
                # possible_layers = {
                #    "channel": [0, 1, 3, 4, 5],
                #    "timepoint": [0, 1],
                # }
                # suffix_with_pims_open_kwargs_per_layer = {
                #    "__channel0_timepoint0", {"channel": 0, "timepoint": 0},
                #    "__channel0_timepoint1", {"channel": 0, "timepoint": 1},
                #    "__channel0_timepoint2", {"channel": 0, "timepoint": 2},
                #    …,
                #    "__channel1_timepoint0", {"channel": 1, "timepoint": 0},
                #    …,
                # }
                suffix_with_pims_open_kwargs_per_layer = {
                    "__" + "_".join(f"{k}{v}" for k, v in sorted(pairs)): dict(pairs)
                    for pairs in product(
                        *(
                            [(key, value) for value in values]
                            for key, values in possible_layers.items()
                        )
                    )
                }
            else:
                # initialize PimsImages as above, with normal layer name
                suffix_with_pims_open_kwargs_per_layer = {"": {}}
                warnings.warn(
                    f"[INFO] There are dimensions beyond channels and xyz which cannot be read: {possible_layers}. "
                    "Defaulting to the first one. "
                    "Please set allow_multiple_layers=True if all of them should be written to different layers, "
                    "or set specific values as arguments.",
                )
        else:
            # initialize PimsImages as above, with normal layer name
            suffix_with_pims_open_kwargs_per_layer = {"": {}}
        first_layer = None
        add_layer_kwargs = {}
        if category == "segmentation":
            add_layer_kwargs["largest_segment_id"] = 0
        if len(suffix_with_pims_open_kwargs_per_layer) > max_layers:
            warnings.warn(
                f"[INFO] Limiting the number of added layers to {max_layers} out of {len(suffix_with_pims_open_kwargs_per_layer)}. "
                + "Please increase `max_layers` if you want more layers to be added.",
            )
        for _, (
            layer_name_suffix,
            pims_open_kwargs,
        ) in zip(range(max_layers), suffix_with_pims_open_kwargs_per_layer.items()):
            # If pims_open_kwargs is empty there's no need to re-open the images:
            if len(pims_open_kwargs) > 0:
                # Set parameters from this method as default
                # if they are not part of the kwargs per layer:
                pims_open_kwargs.setdefault("timepoint", timepoint)  # type: ignore
                pims_open_kwargs.setdefault("channel", channel)  # type: ignore
                pims_open_kwargs.setdefault("czi_channel", czi_channel)  # type: ignore
                pims_image_sequence = pims_images.PimsImages(
                    images,
                    swap_xy=swap_xy,
                    flip_x=flip_x,
                    flip_y=flip_y,
                    flip_z=flip_z,
                    use_bioformats=use_bioformats,
                    is_segmentation=category == "segmentation",
                    **pims_open_kwargs,
                )
            if dtype is None:
                current_dtype = np.dtype(pims_image_sequence.dtype)
                if current_dtype.byteorder == ">":
                    current_dtype = current_dtype.newbyteorder("<")
            else:
                current_dtype = np.dtype(dtype)
            layer = self.add_layer(
                layer_name=layer_name + layer_name_suffix,
                category=category,
                data_format=data_format,
                dtype_per_channel=current_dtype,
                num_channels=pims_image_sequence.num_channels,
                **add_layer_kwargs,  # type: ignore[arg-type]
            )

            expected_bbox = pims_image_sequence.expected_bbox

            # When the expected bbox is 2D the chunk_shape is set to 2D too.
            if expected_bbox.get_shape("z") == 1 and layer.data_format in (
                DataFormat.Zarr,
                DataFormat.Zarr3,
            ):
                chunk_shape = (
                    DEFAULT_CHUNK_SHAPE.with_z(1)
                    if chunk_shape is None
                    else Vec3Int.from_vec_or_int(chunk_shape)
                )
                shard_shape = _get_shard_shape(
                    chunk_shape=chunk_shape,
                    chunks_per_shard=chunks_per_shard,
                    shard_shape=shard_shape,
                )
                if shard_shape is None:
                    if layer.data_format == DataFormat.Zarr3:
                        shard_shape = DEFAULT_SHARD_SHAPE_FROM_IMAGES.with_z(
                            chunk_shape.z
                        )
                    else:
                        shard_shape = DEFAULT_CHUNK_SHAPE.with_z(chunk_shape.z)
                else:
                    shard_shape = Vec3Int.from_vec_or_int(shard_shape)
            else:
                chunk_shape = (
                    DEFAULT_CHUNK_SHAPE
                    if chunk_shape is None
                    else Vec3Int.from_vec_or_int(chunk_shape)
                )
                shard_shape = _get_shard_shape(
                    chunk_shape=chunk_shape,
                    chunks_per_shard=chunks_per_shard,
                    shard_shape=shard_shape,
                )
                if shard_shape is None:
                    if layer.data_format == DataFormat.Zarr3:
                        shard_shape = DEFAULT_SHARD_SHAPE_FROM_IMAGES
                    elif layer.data_format == DataFormat.Zarr:
                        shard_shape = DEFAULT_CHUNK_SHAPE
                    else:
                        shard_shape = DEFAULT_SHARD_SHAPE
                else:
                    shard_shape = Vec3Int.from_vec_or_int(shard_shape)

            mag = Mag(mag)

            # Setting a large enough bounding box, because the exact bounding box
            # cannot be know a priori all the time. It will be replaced with the
            # correct bounding box after reading through all actual images.
            safe_expected_bbox = expected_bbox.from_mag_to_mag1(mag).offset(topleft)
            safe_size = safe_expected_bbox.size.with_replaced(
                safe_expected_bbox.axes.index("x"), SAFE_LARGE_XY
            ).with_replaced(safe_expected_bbox.axes.index("y"), SAFE_LARGE_XY)
            safe_expected_bbox = safe_expected_bbox.with_size(safe_size)
            layer.bounding_box = safe_expected_bbox

            mag_view = layer.add_mag(
                mag=mag,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                compress=compress,
            )

            if batch_size is None:
                if compress or (
                    layer.data_format in (DataFormat.Zarr3, DataFormat.Zarr)
                ):
                    # if data is compressed or dataformat is zarr, parallel write access
                    # to a shard leads to corrupted data, the batch size must be aligned
                    # with the shard size
                    batch_size = mag_view.info.shard_shape.z
                else:
                    # in uncompressed wkw only writing to the same chunk is problematic
                    batch_size = mag_view.info.chunk_shape.z
            elif compress or (layer.data_format in (DataFormat.Zarr3, DataFormat.Zarr)):
                assert batch_size % mag_view.info.shard_shape.z == 0, (
                    f"batch_size {batch_size} must be divisible by z shard-size {mag_view.info.shard_shape.z} when creating compressed layers"
                )
            else:
                assert batch_size % mag_view.info.chunk_shape.z == 0, (
                    f"batch_size {batch_size} must be divisible by z chunk-size {mag_view.info.chunk_shape.z}"
                )

            func_per_chunk = named_partial(
                pims_image_sequence.copy_to_view,
                mag_view=mag_view,
                dtype=current_dtype,
            )

            if (
                additional_axes := set(layer.bounding_box.axes).difference(
                    "x", "y", "z"
                )
            ) and layer.data_format == DataFormat.WKW:
                if all(
                    layer.bounding_box.get_shape(axis) == 1 for axis in additional_axes
                ):
                    warnings.warn(
                        f"[INFO] The data has additional axes {additional_axes}, but they are all of size 1. "
                        + "These axes are not stored in the layer."
                    )
                    layer.bounding_box = BoundingBox.from_ndbbox(layer.bounding_box)
                else:
                    raise RuntimeError(
                        f"WKW datasets only support x, y, z axes, got {additional_axes}. Please use `data_format='zarr3'` instead."
                    )

            buffered_slice_writer_shape = layer.bounding_box.size_xyz.with_z(batch_size)
            args = list(
                layer.bounding_box.chunk(
                    buffered_slice_writer_shape,
                    Vec3Int(1, 1, batch_size),
                )
            )

            with warnings.catch_warnings():
                # We need to catch and ignore a warning here about comparing persisted properties.
                # The problem is that there is an `axisOrder` property that goes in the datasource-properties.json
                # file, but is only stored in a mag object. However, at first we don't have any mags
                # so there is no place to store them. When we add the first mag and update the properties,
                # there is a check that reads the properties first and compares them to the current state.
                # This check fails because the axisOrder property hasn't been stored in the properties file.
                # It is safe to ignore this warning because it is only an intermediate problem in this function.
                # At the end of this function, the properties are complete and consistent.
                warnings.filterwarnings(
                    "ignore",
                    message=".* properties changed in a way that they are not comparable anymore. Most likely the bounding box naming or axis order changed.*",
                    category=UserWarning,
                    module="webknossos",
                )
                with get_executor_for_args(None, executor) as executor:
                    shapes_and_max_ids = wait_and_ensure_success(
                        executor.map_to_futures(func_per_chunk, args),
                        executor=executor,
                        progress_desc=f"Creating layer [bold blue]{layer.name}[/bold blue] from images",
                    )
                shapes, max_ids = zip(*shapes_and_max_ids)
                if category == "segmentation":
                    max_id = max(max_ids)
                    cast(SegmentationLayer, layer).largest_segment_id = max_id
                layer.bounding_box = layer.bounding_box.with_size_xyz(
                    Vec3Int(
                        pims_images.dimwise_max(shapes)
                        + (layer.bounding_box.get_shape("z"),)
                    )
                    * mag.to_vec3_int().with_z(1)
                )
            if expected_bbox != layer.bounding_box:
                warnings.warn(
                    "[WARNING] Some images are larger than expected, smaller slices are padded with zeros now. "
                    + f"New bbox is {layer.bounding_box}, expected {expected_bbox}."
                )

            # Check if category of layer is set correctly
            try:
                if not user_set_category:
                    # When the category is not set by the user, we use a very simple heuristic to guess the category
                    # based on the file name of the input images. After loading the images, we check if the guessed
                    # category might be wrong and adjust it if necessary. This second heuristic is based on the
                    # pixel data of the images
                    guessed_category = guess_category_from_view(layer.get_finest_mag())
                    if guessed_category != layer.category:
                        new_layer_properties: LayerProperties
                        if guessed_category == SEGMENTATION_CATEGORY:
                            logger.info("The layer category is set to segmentation.")
                            new_layer_properties = SegmentationLayerProperties(
                                **(
                                    attr.asdict(layer._properties, recurse=False)
                                ),  # use all attributes from LayerProperties
                                largest_segment_id=int(max(max_ids)),
                            )
                            new_layer_properties.category = SEGMENTATION_CATEGORY
                            self._layers[layer.name] = SegmentationLayer(
                                self, new_layer_properties, read_only=False
                            )
                        else:
                            logger.info("The layer category is set to color.")
                            _properties = attr.asdict(layer._properties, recurse=False)
                            _properties.pop("largest_segment_id", None)
                            _properties.pop("mappings", None)

                            new_layer_properties = LayerProperties(**_properties)
                            new_layer_properties.category = COLOR_CATEGORY
                            self._layers[layer.name] = Layer(
                                self, new_layer_properties, read_only=False
                            )
                        self._properties.update_for_layer(
                            layer.name, new_layer_properties
                        )
                        self._save_dataset_properties()

            except Exception:
                # The used heuristic was not able to guess the layer category, the previous value is kept
                pass
            if first_layer is None:
                first_layer = layer
        assert first_layer is not None
        return first_layer

    def write_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        data: np.ndarray,  # in specified mag
        *,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        downsample: bool = True,
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: Vec3IntLike | int | None = None,
        axes: Iterable[str] | None = None,
        absolute_offset: Vec3IntLike | VecIntLike | None = None,  # in mag1
        mag: MagLike = Mag(1),
    ) -> Layer:
        """Write a numpy array to a new layer and downsample.

        Args:
            layer_name: Name of the new layer.
            category: Category of the new layer.
            data: The data to write.
            data_format: Format to store the data. Defaults to zarr3.
            downsample: Whether to downsample the data. Defaults to True.
            chunk_shape: Shape of chunks for storage. Recommended (32,32,32) or (64,64,64). Defaults to (32,32,32).
            shard_shape: Shape of shards for storage. Must be a multiple of chunk_shape. If specified, chunks_per_shard must not be specified. Defaults to (1024, 1024, 1024).
            chunks_per_shard: Deprecated, use shard_shape. Number of chunks per shards. If specified, shard_shape must not be specified.
            axes: The axes of the data for non-3D data.
            absolute_offset: The offset of the data. Specified in Mag 1.
            mag: Magnification to write the data at.
        """
        mag = Mag(mag)
        bbox, num_channels = derive_nd_bounding_box_from_shape(
            data.shape, axes=axes, absolute_offset=absolute_offset
        )
        mag1_bbox = bbox.with_size_xyz(bbox.size_xyz * mag.to_vec3_int())
        layer = self.add_layer(
            layer_name,
            category,
            data_format=data_format,
            num_channels=num_channels,
            dtype_per_channel=data.dtype,
            bounding_box=mag1_bbox,
        )

        with warnings.catch_warnings():
            # For n-d datasets, the `axisOrder` property is stored with mags.
            # At this point, we don't have any mags yet, so we can't compare the persisted properties.
            warnings.filterwarnings(
                "ignore",
                message=".* properties changed in a way that they are not comparable anymore. Most likely the bounding box naming or axis order changed.*",
                category=UserWarning,
                module="webknossos",
            )
            mag_view = layer.add_mag(
                mag,
                chunk_shape=chunk_shape,
                chunks_per_shard=chunks_per_shard,
                shard_shape=shard_shape,
                compress=True,
            )
        mag_view.write(data, absolute_bounding_box=layer.bounding_box)

        if downsample:
            layer.downsample()

        return layer

    def delete_layer(self, layer_name: str) -> None:
        """Delete a layer from the dataset.

        Removes the layer's data and metadata from disk completely.
        This deletes both the datasource-properties.json entry and all
        data files for the layer.

        Args:
            layer_name: Name of layer to delete

        Raises:
            IndexError: If no layer with the given name exists
            RuntimeError: If dataset is read-only

        Examples:
            ```
            ds.delete_layer("old_layer")
            print("Remaining layers:", list(ds.layers))
            ```
        """

        self._ensure_writable()

        if layer_name not in self.layers.keys():
            raise IndexError(
                f"Removing layer {layer_name} failed. There is no layer with this name"
            )
        layer_path = self._layers[layer_name].path
        del self._layers[layer_name]
        self._properties.data_layers = [
            layer for layer in self._properties.data_layers if layer.name != layer_name
        ]
        # delete files on disk
        # rmtree does not recurse into linked dirs, but removes the link
        rmtree(layer_path)
        self._save_dataset_properties()

    def add_copy_layer(
        self,
        foreign_layer: str | PathLike | UPath | Layer | RemoteLayer,
        new_layer_name: str | None = None,
        *,
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: Vec3IntLike | int | None = None,
        data_format: str | DataFormat | None = None,
        compress: bool | None = None,
        exists_ok: bool = False,
        executor: Executor | None = None,
        with_attachments: bool = True,
    ) -> Layer:
        """Deprecated. Use `Dataset.add_layer_as_copy` instead."""
        warn_deprecated("add_copy_layer", "add_layer_as_copy")
        return self.add_layer_as_copy(
            foreign_layer,
            new_layer_name,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            chunks_per_shard=chunks_per_shard,
            data_format=data_format,
            compress=compress,
            exists_ok=exists_ok,
            executor=executor,
            with_attachments=with_attachments,
        )

    def add_layer_as_copy(
        self,
        foreign_layer: str | PathLike | UPath | Layer | RemoteLayer,
        new_layer_name: str | None = None,
        *,
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: Vec3IntLike | int | None = None,
        data_format: str | DataFormat | None = None,
        compress: bool | Zarr3Config | None = None,
        exists_ok: bool = False,
        executor: Executor | None = None,
        with_attachments: bool = True,
    ) -> Layer:
        """Copy layer from another dataset to this one.

        Creates a new layer in this dataset by copying data and metadata from
        a layer in another dataset.

        Args:
            foreign_layer: Layer to copy (path or Layer object)
            new_layer_name: Optional name for the new layer, uses original name if None
            chunk_shape: Optional shape of chunks for storage
            shard_shape: Optional shape of shards for storage
            chunks_per_shard: Deprecated, use shard_shape. Optional number of chunks per shard
            data_format: Optional format to store copied data ('wkw', 'zarr', etc.)
            compress: Optional whether to compress copied data
            exists_ok: Whether to overwrite existing layers
            executor: Optional executor for parallel copying

        Returns:
            Layer: The newly created copy of the layer

        Raises:
            IndexError: If target layer name already exists
            RuntimeError: If dataset is read-only

        Examples:
            Copy layer keeping same name:
            ```
            other_ds = Dataset.open("other/dataset")
            copied = ds.add_layer_as_copy(other_ds.get_layer("color"))
            ```

            Copy with new name:
            ```
            copied = ds.add_layer_as_copy(
                other_ds.get_layer("color"),
                new_layer_name="color_copy",
                compress=True
            )
            ```
        """
        self._ensure_writable()
        foreign_layer = Layer._ensure_layer(foreign_layer)

        if new_layer_name is None:
            new_layer_name = foreign_layer.name
        else:
            _validate_layer_name(new_layer_name)

        if exists_ok:
            layer = self.get_or_add_layer(
                new_layer_name,
                category=foreign_layer.category,
                dtype_per_channel=foreign_layer.dtype_per_channel,
                num_channels=foreign_layer.num_channels,
                data_format=data_format or foreign_layer.data_format,
                largest_segment_id=foreign_layer._get_largest_segment_id_maybe(),
                bounding_box=foreign_layer.bounding_box,
            )
        else:
            if new_layer_name in self.layers.keys():
                raise IndexError(
                    f"Cannot copy {foreign_layer}. This dataset already has a layer called {new_layer_name}."
                )
            layer = self.add_layer(
                new_layer_name,
                category=foreign_layer.category,
                dtype_per_channel=foreign_layer.dtype_per_channel,
                num_channels=foreign_layer.num_channels,
                data_format=data_format or foreign_layer.data_format,
                largest_segment_id=foreign_layer._get_largest_segment_id_maybe(),
                bounding_box=foreign_layer.bounding_box,
            )

        for mag_view in foreign_layer.mags.values():
            progress_desc = (
                f"Copying {mag_view.layer.name}/{mag_view.mag.to_layer_name()}"
            )

            layer.add_mag_as_copy(
                mag_view,
                extend_layer_bounding_box=False,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                chunks_per_shard=chunks_per_shard,
                compress=compress,
                exists_ok=exists_ok,
                executor=executor,
                progress_desc=progress_desc,
            )

        if (
            with_attachments
            and isinstance(layer, SegmentationLayer)
            and isinstance(foreign_layer, SegmentationLayer | RemoteSegmentationLayer)
        ):
            for attachment in foreign_layer.attachments:
                layer.attachments.add_attachment_as_copy(attachment)

        return layer

    def add_symlink_layer(
        self,
        foreign_layer: str | PathLike | UPath | Layer,
        new_layer_name: str | None = None,
        *,
        make_relative: bool = False,
    ) -> Layer:
        """Deprecated. Use `Dataset.add_layer_as_ref` instead.

        Create symbolic link to layer from another dataset.

        Instead of copying data, creates a symbolic link to the original layer's data and copies
        only the layer metadata. Changes to the original layer's properties, e.g. bounding box, afterwards won't
        affect this dataset and vice-versa.

        Args:
            foreign_layer: Layer to link to (path or Layer object)
            make_relative: Whether to create relative symlinks
            new_layer_name: Optional name for the linked layer, uses original name if None

        Returns:
            Layer: The newly created symbolic link layer

        Raises:
            IndexError: If target layer name already exists
            AssertionError: If trying to create symlinks in/to remote datasets
            RuntimeError: If dataset is read-only

        Examples:
            ```
            other_ds = Dataset.open("other/dataset")
            linked = ds.add_symlink_layer(
                other_ds.get_layer("color"),
                make_relative=True
            )
            ```

        Note:
            Only works with local file systems, cannot link remote datasets or
            create symlinks in remote datasets.
        """

        self._ensure_writable()
        warnings.warn(
            "Using symlinks is deprecated and will be removed in a future version. "
            + "Use `add_layer_as_ref` instead, which adds the mags and attachments of the layer as references to this dataset.",
            DeprecationWarning,
            stacklevel=2,
        )

        maybe_remote_foreign_layer = Layer._ensure_layer(foreign_layer)
        if not isinstance(maybe_remote_foreign_layer, Layer):
            raise TypeError(f"Cannot create symlink to remote layer {foreign_layer}.")

        foreign_layer = maybe_remote_foreign_layer

        if new_layer_name is None:
            new_layer_name = foreign_layer.name

        if new_layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer}. This dataset already has a layer called {new_layer_name}."
            )
        foreign_layer_path = foreign_layer.path

        assert is_fs_path(self.path), (
            f"Cannot create symlinks in remote dataset {self.path}"
        )

        assert is_fs_path(foreign_layer_path) and foreign_layer_path is not None, (
            f"Cannot create symlink to remote layer {foreign_layer_path}"
        )

        foreign_layer_symlink_path = (
            UPath(relpath(foreign_layer_path, self.path))
            if make_relative
            else foreign_layer_path.resolve()
        )

        new_layer_path = self.path / new_layer_name
        new_layer_path.symlink_to(foreign_layer_symlink_path)
        new_layer_properties = copy.deepcopy(foreign_layer._properties)
        new_layer_properties.name = new_layer_name

        # Add correct paths to mag properties
        for foreign_mag in foreign_layer.mags.values():
            mag_prop = next(
                m for m in new_layer_properties.mags if m.mag == foreign_mag.mag
            )
            if is_fs_path(foreign_mag.path):
                mag_prop.path = (
                    UPath(relpath(foreign_mag.path.resolve(), self.path))
                    if make_relative
                    else foreign_mag.path.resolve()
                ).as_posix()
            else:
                mag_prop.path = dump_path(foreign_mag.path, self.resolved_path)

        if (
            isinstance(new_layer_properties, SegmentationLayerProperties)
            and new_layer_properties.attachments is not None
        ):
            for attachment in new_layer_properties.attachments:
                old_path = UPath(attachment.path)
                if is_fs_path(old_path):
                    if not old_path.is_absolute():
                        old_path = (
                            foreign_layer.dataset.resolved_path / old_path
                        ).resolve()
                    assert is_fs_path(old_path)
                    attachment.path = (
                        UPath(relpath(old_path, self.path))
                        if make_relative
                        else old_path.resolve()
                    ).as_posix()

        self._properties.data_layers += [new_layer_properties]
        self._layers[new_layer_name] = self._initialize_layer_from_properties(
            new_layer_properties, read_only=True
        )

        self._save_dataset_properties()
        return self.layers[new_layer_name]

    def add_remote_layer(
        self,
        foreign_layer: str | PathLike | UPath | Layer | RemoteLayer,
        new_layer_name: str | None = None,
    ) -> Layer:
        """Deprecated. Use `Dataset.add_layer_as_ref` instead."""
        warn_deprecated("add_remote_layer", "add_layer_as_ref")
        return self.add_layer_as_ref(foreign_layer, new_layer_name)

    def add_layer_as_ref(
        self,
        foreign_layer: str | PathLike | UPath | Layer | RemoteLayer,
        new_layer_name: str | None = None,
    ) -> Layer:
        """Add a layer from another dataset by reference.

        Creates a layer that references data from a remote dataset. The image data
        will be streamed on-demand when accessed.

        Args:
            foreign_layer: Foreign layer to add (path or Layer object)
            new_layer_name: Optional name for the new layer, uses original name if None

        Returns:
            Layer: The newly created remote layer referencing the foreign data

        Raises:
            IndexError: If target layer name already exists
            AssertionError: If trying to add non-remote layer or same origin dataset
            RuntimeError: If dataset is read-only

        Examples:
            ```
            ds = Dataset.open("other/dataset")
            remote_ds = RemoteDataset.open("my_dataset", "my_org_id")
            new_layer = ds.add_layer_as_ref(
                remote_ds.get_layer("color")
            )
            ```

        Note:
            Changes to the original layer's properties afterwards won't affect this dataset.
            Data is only referenced, not copied.
        """

        self._ensure_writable()
        foreign_layer = Layer._ensure_layer(foreign_layer)

        if new_layer_name is None:
            new_layer_name = foreign_layer.name
        else:
            _validate_layer_name(new_layer_name)

        if new_layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot add foreign layer {foreign_layer}. This dataset already has a layer called {new_layer_name}."
            )
        assert not foreign_layer.dataset == self, (
            "Cannot add layer with the same origin dataset as foreign layer"
        )

        new_layer = self.add_layer_like(foreign_layer, new_layer_name)
        for mag_view in foreign_layer.mags.values():
            new_layer.add_mag_as_ref(mag_view, extend_layer_bounding_box=False)

        # reference-copy all attachments
        if isinstance(
            foreign_layer, SegmentationLayer | RemoteSegmentationLayer
        ) and isinstance(new_layer, SegmentationLayer):
            for attachment in foreign_layer.attachments:
                new_layer.attachments.add_attachment_as_ref(attachment)
        return new_layer

    def add_fs_copy_layer(
        self,
        foreign_layer: str | PathLike | UPath | Layer,
        new_layer_name: str | None = None,
    ) -> Layer:
        """Deprecated. File-based copy is automatically used in `Dataset.add_layer_as_copy`.

        Copies the files at `foreign_layer` which belongs to another dataset
        to the current dataset via the filesystem. Additionally, the relevant
        information from the `datasource-properties.json` of the other dataset
        are copied too. If new_layer_name is None, the name of the foreign
        layer is used.
        """
        warn_deprecated("add_fs_copy_layer", "add_layer_as_copy")
        self._ensure_writable()
        maybe_remote_foreign_layer = Layer._ensure_layer(foreign_layer)
        if not isinstance(maybe_remote_foreign_layer, Layer):
            raise TypeError(f"Cannot create symlink to remote layer {foreign_layer}.")

        foreign_layer = maybe_remote_foreign_layer
        if new_layer_name is None:
            new_layer_name = foreign_layer.name

        if new_layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot copy {foreign_layer}. This dataset already has a layer called {new_layer_name}."
            )

        copytree(foreign_layer.path, self.path / new_layer_name)
        new_layer_properties = copy.deepcopy(foreign_layer._properties)
        new_layer_properties.name = new_layer_name

        if (
            isinstance(new_layer_properties, SegmentationLayerProperties)
            and new_layer_properties.attachments is not None
        ):
            for attachment in new_layer_properties.attachments:
                old_path = UPath(attachment.path)
                if is_fs_path(old_path):
                    assert isinstance(old_path, UPath)  # for mypy
                    if not old_path.is_absolute():
                        old_path = (
                            foreign_layer.dataset.resolved_path / old_path.as_posix()
                        ).resolve()
                    else:
                        old_path = old_path.resolve()
                    # attachment has been a foreign attachment to the foreign layer
                    # therefore it will not be copied
                    if foreign_layer.resolved_path not in old_path.parents:
                        attachment.path = old_path.as_posix()

        self._properties.data_layers += [new_layer_properties]
        self._layers[new_layer_name] = self._initialize_layer_from_properties(
            new_layer_properties, read_only=False
        )

        self._save_dataset_properties()
        return self.layers[new_layer_name]

    def copy_dataset(
        self,
        new_dataset_path: str | PathLike | UPath,
        *,
        voxel_size: tuple[float, float, float] | None = None,
        chunk_shape: Vec3IntLike | int | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        chunks_per_shard: Vec3IntLike | int | None = None,
        data_format: str | DataFormat | None = None,
        compress: bool | None = None,
        exists_ok: bool = False,
        executor: Executor | None = None,
        voxel_size_with_unit: VoxelSize | None = None,
        layers_to_ignore: Iterable[str] | None = None,
    ) -> "Dataset":
        """
        Creates an independent copy of the dataset with all layers at a new location.
        Data storage parameters can be customized for the copied dataset.

        Args:
            new_dataset_path: Path where new dataset should be created
            voxel_size: Optional tuple of floats (x,y,z) specifying voxel size in nanometers
            chunk_shape: Optional shape of chunks for data storage
            shard_shape: Optional shape of shards for data storage
            chunks_per_shard: Deprecated, use shard_shape. Optional number of chunks per shard
            data_format: Optional format to store data ('wkw', 'zarr', 'zarr3')
            compress: Optional whether to compress data
            exists_ok: Whether to overwrite existing datasets and layers
            executor: Optional executor for parallel copying
            voxel_size_with_unit: Optional voxel size specification with units
            layers_to_ignore: List of layer names to exclude from the copy

        Returns:
            Dataset: The newly created copy

        Raises:
            AssertionError: If trying to copy WKW layers to remote dataset

        Examples:
            Basic copy:
                ```
                copied = ds.copy_dataset("path/to/copy")
                ```

            Copy with different storage:
                ```
                copied = ds.copy_dataset(
                    "path/to/copy",
                    data_format="zarr",
                    compress=True
                )
                ```

        Note:
            WKW layers can only be copied to datasets on local file systems.
            For remote datasets, use data_format='zarr3'.
        """

        new_dataset_path = UPath(new_dataset_path)

        if data_format == DataFormat.WKW:
            assert is_fs_path(new_dataset_path), (
                "Cannot create WKW-based remote datasets. Use `data_format='zarr3'` instead."
            )
        if data_format is None and any(
            layer.data_format == DataFormat.WKW for layer in self.layers.values()
        ):
            assert is_fs_path(new_dataset_path), (
                "Cannot create WKW layers in remote datasets. Use explicit `data_format='zarr3'`."
            )

        if voxel_size_with_unit is None:
            if voxel_size is None:
                voxel_size_with_unit = self.voxel_size_with_unit
            else:
                voxel_size_with_unit = VoxelSize(voxel_size)
        new_dataset = Dataset(
            new_dataset_path,
            voxel_size_with_unit=voxel_size_with_unit,
            exist_ok=exists_ok,
        )
        new_dataset.default_view_configuration = self.default_view_configuration

        with get_executor_for_args(None, executor) as executor:
            for layer in self.layers.values():
                if layers_to_ignore is not None and layer.name in layers_to_ignore:
                    continue
                new_dataset.add_layer_as_copy(
                    layer,
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    chunks_per_shard=chunks_per_shard,
                    data_format=data_format,
                    compress=compress,
                    exists_ok=exists_ok,
                    executor=executor,
                )
        new_dataset._save_dataset_properties()
        return new_dataset

    def fs_copy_dataset(
        self,
        new_dataset_path: str | PathLike | UPath,
        *,
        exists_ok: bool = False,
        layers_to_ignore: Iterable[str] | None = None,
    ) -> "Dataset":
        """Deprecated. File-based copy is automatically used by `Dataset.copy_dataset`.

        Creates an independent copy of the dataset with all layers at a new location.

        This method copies the files of the dataset as is and, therefore, might be faster than Dataset.copy_dataset, which decodes and encodes all the data.
        If you wish to change the data storage parameters, use Dataset.copy_dataset.

        Args:
            new_dataset_path: Path where new dataset should be created
            exists_ok: Whether to overwrite existing datasets and layers
            layers_to_ignore: List of layer names to exclude from the copy

        Returns:
            Dataset: The newly created copy

        Raises:
            AssertionError: If trying to copy WKW layers to remote dataset

        Examples:
            Basic copy:
                ```
                copied = ds.fs_copy_dataset("path/to/copy")
                ```

        Note:
            WKW layers can only be copied to datasets on local file systems.
        """
        warn_deprecated("fs_copy_dataset", "copy_dataset")

        new_dataset_path = UPath(new_dataset_path)

        if any(layer.data_format == DataFormat.WKW for layer in self.layers.values()):
            assert is_fs_path(new_dataset_path), (
                "Cannot create WKW layers in remote datasets. Use `Dataset.copy_dataset` with `data_format='zarr3'`."
            )

        new_dataset = Dataset(
            new_dataset_path,
            voxel_size_with_unit=self.voxel_size_with_unit,
            exist_ok=exists_ok,
        )
        new_dataset.default_view_configuration = self.default_view_configuration

        for layer in self.layers.values():
            if layers_to_ignore is not None and layer.name in layers_to_ignore:
                continue
            new_layer = new_dataset.add_layer_like(layer, layer.name)
            for mag_view in layer.mags.values():
                new_mag = new_layer.add_mag(
                    mag_view.mag,
                    chunk_shape=mag_view.info.chunk_shape,
                    shard_shape=mag_view.info.shard_shape,
                    compress=mag_view.info.compression_mode,
                )
                copytree(mag_view.path, new_mag.path)
            if isinstance(layer, SegmentationLayer) and isinstance(
                new_layer, SegmentationLayer
            ):
                for attachment in layer.attachments:
                    new_layer.attachments.add_attachment_as_copy(attachment)
        new_dataset._save_dataset_properties()
        return new_dataset

    def shallow_copy_dataset(
        self,
        new_dataset_path: str | PathLike | UPath,
        *,
        name: str | None = None,
        layers_to_ignore: Iterable[str] | None = None,
        make_relative: bool | None = None,  # deprecated
    ) -> "Dataset":
        """Create a new dataset that contains references to the layers, mags and attachments of another dataset.

        Useful for creating alternative views or exposing datasets to WEBKNOSOSS.

        Args:
            new_dataset_path: Path where new dataset should be created
            name: Optional name for the new dataset, uses original name if None
            layers_to_ignore: Optional iterable of layer names to exclude
            executor: Optional executor for copy operations

        Returns:
            Dataset: The newly created dataset with linked layers

        Raises:
            RuntimeError: If dataset is read-only

        Examples:
            Basic shallow copy:
                ```
                linked = ds.shallow_copy_dataset("path/to/link")
                ```

            With relative links excluding layers:
                ```
                linked = ds.shallow_copy_dataset(
                    "path/to/link",
                    make_relative=True,
                    layers_to_ignore=["temp_layer"]
                )
                ```
        """
        if make_relative is not None:
            warnings.warn(
                "make_relative is deprecated and has no utility anymore, because shallow_copy_dataset does not use symlinks anymore.",
                DeprecationWarning,
                stacklevel=2,
            )

        new_dataset_path = UPath(new_dataset_path)
        new_dataset = Dataset(
            new_dataset_path,
            voxel_size_with_unit=self.voxel_size_with_unit,
            name=name or self.name,
            exist_ok=False,
        )
        new_dataset.default_view_configuration = self.default_view_configuration

        for layer_name, layer in self.layers.items():
            if layers_to_ignore is not None and layer_name in layers_to_ignore:
                continue
            new_dataset.add_layer_as_ref(layer, layer_name)

        return new_dataset

    def compress(
        self,
        *,
        executor: Executor | None = None,
    ) -> None:
        """Compress all uncompressed magnifications in-place.

        Compresses the data of all magnification levels that aren't already compressed,
        for all layers in the dataset.

        Args:
            executor: Optional executor for parallel compression

        Raises:
            RuntimeError: If dataset is read-only

        Examples:
            ```
            ds.compress()
            ```

        Note:
            If data is already compressed, this will have no effect.

        """
        for layer in self.layers.values():
            for mag in layer.mags.values():
                if not mag._is_compressed():
                    mag.compress(executor=executor)

    def downsample(
        self,
        *,
        sampling_mode: SamplingModes = SamplingModes.ANISOTROPIC,
        coarsest_mag: Mag | None = None,
        interpolation_mode: str = "default",
        compress: bool | Zarr3Config = True,
        executor: Executor | None = None,
    ) -> None:
        """Generate downsampled magnifications for all layers.

        Creates lower resolution versions (coarser magnifications) of all layers that are not
        yet downsampled, up to the specified coarsest magnification.

        Args:
            sampling_mode: Strategy for downsampling (e.g. ANISOTROPIC, MAX)
            coarsest_mag: Optional maximum/coarsest magnification to generate
            interpolation_mode: Interpolation method to use. Defaults to "default" (= "mode" for segmentation, "median" for color).
            compress: Whether to compress generated magnifications. For Zarr3 datasets, codec configuration and chunk key encoding may also be supplied. Defaults to True.
            executor: Optional executor for parallel processing

        Raises:
            RuntimeError: If dataset is read-only

        Examples:
            Basic downsampling:
                ```
                ds.downsample()
                ```

            With custom parameters:
                ```
                ds.downsample(
                    sampling_mode=SamplingModes.ANISOTROPIC,
                    coarsest_mag=Mag(8),
                )
                ```

        Note:
            - ANISOTROPIC sampling creates anisotropic downsampling until dataset is isotropic
            - Other modes like MAX, CONSTANT etc create regular downsampling patterns
            - If magnifications already exist they will not be regenerated
        """
        for layer in self.layers.values():
            layer.downsample(
                coarsest_mag=coarsest_mag,
                sampling_mode=sampling_mode,
                interpolation_mode=interpolation_mode,
                compress=compress,
                executor=executor,
            )

    def _get_layer_by_category(self, category: LayerCategoryType) -> Layer:
        assert category == COLOR_CATEGORY or category == SEGMENTATION_CATEGORY

        layers = [layer for layer in self.layers.values() if category == layer.category]

        if len(layers) == 1:
            return layers[0]
        elif len(layers) == 0:
            raise IndexError(
                f"Failed to get segmentation layer: There is no {category} layer."
            )
        else:
            raise IndexError(
                f"Failed to get segmentation layer: There are multiple {category} layer."
            )
