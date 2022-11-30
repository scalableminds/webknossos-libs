import copy
import inspect
import json
import logging
import re
import warnings
from argparse import Namespace
from contextlib import nullcontext
from enum import Enum, unique
from itertools import product
from os import PathLike
from os.path import relpath
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import attr
import numpy as np
from boltons.typeutils import make_sentinel
from cluster_tools import Executor
from natsort import natsort_keygen
from upath import UPath

from ..geometry.vec3_int import Vec3Int, Vec3IntLike
from ._array import ArrayException, ArrayInfo, BaseArray, DataFormat
from .remote_dataset_registry import RemoteDatasetRegistry
from .sampling_modes import SamplingModes

if TYPE_CHECKING:
    import pims
    from ..client._generated.models import DatasetInfoResponse200
    from ..client._upload_dataset import LayerToLink
    from ..administration.user import Team

from ..geometry import BoundingBox, Mag
from ..utils import (
    copy_directory_with_symlinks,
    copytree,
    get_executor_for_args,
    is_fs_path,
    named_partial,
    rmtree,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._utils.from_images import guess_if_segmentation_path
from ._utils.infer_bounding_box_existing_files import infer_bounding_box_existing_files
from .layer import (
    DTypeLike,
    Layer,
    SegmentationLayer,
    _dtype_per_channel_to_element_class,
    _dtype_per_layer_to_dtype_per_channel,
    _get_sharding_parameters,
    _normalize_dtype_per_channel,
    _normalize_dtype_per_layer,
)
from .layer_categories import COLOR_CATEGORY, SEGMENTATION_CATEGORY, LayerCategoryType
from .properties import (
    DatasetProperties,
    DatasetViewConfiguration,
    LayerProperties,
    SegmentationLayerProperties,
    _extract_num_channels,
    _properties_floating_type_to_python_type,
    dataset_converter,
)
from .view import _BLOCK_ALIGNMENT_WARNING

logger = logging.getLogger(__name__)

DEFAULT_BIT_DEPTH = 8
DEFAULT_DATA_FORMAT = DataFormat.WKW
PROPERTIES_FILE_NAME = "datasource-properties.json"
ZGROUP_FILE_NAME = ".zgroup"
ZATTRS_FILE_NAME = ".zattrs"

_DATASET_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/datasets/"
    + r"(?P<organization_id>[^/]*)/(?P<dataset_name>[^/]*)(/(view)?)?"
    + r"(\?token=(?P<sharing_token>[^#]*))?"
)


def _find_array_info(layer_path: Path) -> Optional[ArrayInfo]:
    for f in layer_path.iterdir():
        if f.is_dir():
            try:
                array = BaseArray.open(f)
                return array.info
            except ArrayException:
                pass
    return None


_UNSET = make_sentinel("UNSET", var_name="_UNSET")

_UNSPECIFIED_SCALE_FROM_OPEN = make_sentinel(
    "_UNSPECIFIED_SCALE_FROM_OPEN", var_name="_UNSPECIFIED_SCALE_FROM_OPEN"
)


class Dataset:
    """
    A dataset is the entry point of the Dataset API.
    An existing dataset on disk can be opened
    or new datasets can be created.

    A dataset stores the data in `.wkw` files on disk with metadata in `datasource-properties.json`.
    The information in those files are kept in sync with the object.

    Each dataset consists of one or more layers (webknossos.dataset.layer.Layer),
    which themselves can comprise multiple magnifications (webknossos.dataset.mag_view.MagView).

    When using `Dataset.open_remote()` an instance of the `RemoteDataset` subclass is returned.
    """

    @unique
    class ConversionLayerMapping(Enum):
        """Strategies for mapping file paths to layers, for use in
        `Dataset.from_images` for the `map_filepath_to_layer_name` argument.

        If none of the strategies fit, the mapping can also be specified by a callable.
        """

        INSPECT_SINGLE_FILE = "inspect_single_file"
        """The first found image file is opened. If it appears to be
        a 2D image, `ENFORCE_LAYER_PER_FOLDER` is used,
        if it appears to be 3D, `ENFORCE_LAYER_PER_FILE` is used.
        This is the default mapping."""

        INSPECT_EVERY_FILE = "inspect_every_file"
        """Like `INSPECT_SINGLE_FILE`, but the strategy
        is determined for each image file separately."""

        ENFORCE_LAYER_PER_FILE = "enforce_layer_per_file"
        """Enforce a new layer per file. This is useful for 2D
        images that should be converted to 2D layers each."""

        ENFORCE_SINGLE_LAYER = "enforce_single_layer"
        """Combines all found files into a single layer. This is only
        useful if all images are 2D."""

        ENFORCE_LAYER_PER_FOLDER = "enforce_layer_per_folder"
        """Combine all files in a folder into one layer."""

        ENFORCE_LAYER_PER_TOPLEVEL_FOLDER = "enforce_layer_per_toplevel_folder"
        """The first folders of the input path are each converted to one layer.
        This might be useful if multiple layers have stacks of 2D images, but
        parts of the stacks are in different folders."""

        def _to_callable(
            self, input_path: Path, input_files: Sequence[Path], use_bioformats: bool
        ) -> Callable[[Path], str]:
            from ._utils.pims_images import has_image_z_dimension

            ConversionLayerMapping = Dataset.ConversionLayerMapping

            if self == ConversionLayerMapping.ENFORCE_LAYER_PER_FILE:
                return str
            elif self == ConversionLayerMapping.ENFORCE_SINGLE_LAYER:
                return lambda p: input_path.name
            elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_FOLDER:
                return (
                    lambda p: input_path.name if p.parent == Path() else str(p.parent)
                )
            elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_TOPLEVEL_FOLDER:
                return lambda p: input_path.name if p.parent == Path() else p.parts[0]
            elif self == ConversionLayerMapping.INSPECT_EVERY_FILE:
                # If a file has z dimensions, it becomes its own layer,
                # if it's 2D, the folder becomes a layer.
                return (
                    lambda p: str(p)
                    if has_image_z_dimension(
                        input_path / p,
                        use_bioformats=use_bioformats,
                        is_segmentation=guess_if_segmentation_path(p),
                    )
                    else input_path.name
                    if p.parent == Path()
                    else str(p.parent)
                )
            elif self == ConversionLayerMapping.INSPECT_SINGLE_FILE:
                # As before, but only a single image is inspected to determine 2D vs 3D.
                if has_image_z_dimension(
                    input_path / input_files[0],
                    use_bioformats=use_bioformats,
                    is_segmentation=guess_if_segmentation_path(input_files[0]),
                ):
                    return str
                else:
                    return (
                        lambda p: input_path.name
                        if p.parent == Path()
                        else str(p.parent)
                    )
            else:
                raise ValueError(f"Got unexpected ConversionLayerMapping value: {self}")

    def __init__(
        self,
        dataset_path: Union[str, PathLike],
        voxel_size: Optional[Tuple[float, float, float]] = None,
        name: Optional[str] = None,
        exist_ok: bool = _UNSET,
        *,
        scale: Optional[Tuple[float, float, float]] = None,
        read_only: bool = False,
    ) -> None:
        """
        Creates a new dataset and the associated `datasource-properties.json`.
        If the dataset already exists and exist_ok is set to True,
        it is opened (the provided voxel_size and name are asserted to match the existing dataset).
        Currently, `exist_ok=True` is the deprecated default and will change in future releases.
        Please use `Dataset.open` if you intend to open an existing dataset and don't want/need the creation behavior.
        `scale` is deprecated, please use `voxel_size` instead.
        """
        if scale is not None:
            assert (
                voxel_size is None
            ), "Cannot use scale and voxel_size, please use only voxel_size."
            warn_deprecated("scale", "voxel_size")
            voxel_size = scale

        self._read_only = read_only

        dataset_path = UPath(dataset_path)

        dataset_existed_already = (
            dataset_path.exists()
            and dataset_path.is_dir()
            and next(dataset_path.iterdir(), None) is not None
        )

        if dataset_existed_already:
            if exist_ok == _UNSET:
                warnings.warn(
                    f"[DEPRECATION] You are creating/opening a dataset at a non-empty folder {dataset_path} without setting exist_ok=True. "
                    + "This will fail in future releases, please supply exist_ok=True explicitly then.",
                    DeprecationWarning,
                )
                exist_ok = True
            if not exist_ok:
                raise RuntimeError(
                    f"Creation of Dataset at {dataset_path} failed, because a non-empty folder already exists at this path."
                )
            assert (
                dataset_path / PROPERTIES_FILE_NAME
            ).is_file(), f"Cannot open Dataset: Could not find {PROPERTIES_FILE_NAME} in non-empty directory {dataset_path}"
        else:
            if read_only:
                raise FileNotFoundError(
                    f"Cannot create read-only dataset, could not find data at {dataset_path}."
                )
            assert (
                not dataset_path.exists() or dataset_path.is_dir()
            ), f"Creation of Dataset at {dataset_path} failed, because a file already exists at this path."
            # Create directories on disk and write datasource-properties.json
            try:
                dataset_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise type(e)(
                    "Creation of Dataset {} failed. ".format(dataset_path) + repr(e)
                )

            # Write empty properties to disk
            assert (
                voxel_size is not None
            ), "When creating a new dataset, the voxel_size must be given, e.g. as Dataset(path, voxel_size=(10, 10, 16.8))"
            name = name or dataset_path.absolute().name
            dataset_properties = DatasetProperties(
                id={"name": name, "team": ""}, scale=voxel_size, data_layers=[]
            )
            with (dataset_path / PROPERTIES_FILE_NAME).open(
                "w", encoding="utf-8"
            ) as outfile:
                json.dump(
                    dataset_converter.unstructure(dataset_properties), outfile, indent=4
                )

        self.path: Path = dataset_path
        self._properties: DatasetProperties = self._load_properties()
        self._last_read_properties = copy.deepcopy(self._properties)

        self._layers: Dict[str, Layer] = {}
        # construct self.layer
        for layer_properties in self._properties.data_layers:
            num_channels = _extract_num_channels(
                layer_properties.num_channels,
                UPath(dataset_path),
                layer_properties.name,
                layer_properties.mags[0].mag
                if len(layer_properties.mags) > 0
                else None,
            )
            layer_properties.num_channels = num_channels

            layer = self._initialize_layer_from_properties(layer_properties)
            self._layers[layer_properties.name] = layer

        if dataset_existed_already:
            if voxel_size is None:
                warnings.warn(
                    "[DEPRECATION] Please always supply the voxel_size when using the constructor Dataset(your_path, voxel_size=your_voxel_size)."
                    + "If you just want to open an existing dataset, please use Dataset.open(your_path).",
                    DeprecationWarning,
                )
            elif voxel_size == _UNSPECIFIED_SCALE_FROM_OPEN:
                pass
            else:
                assert self.voxel_size == tuple(
                    voxel_size
                ), f"Cannot open Dataset: The dataset {dataset_path} already exists, but the voxel_sizes do not match ({self.voxel_size} != {voxel_size})"
            if name is not None:
                assert (
                    self.name == name
                ), f"Cannot open Dataset: The dataset {dataset_path} already exists, but the names do not match ({self.name} != {name})"

    @classmethod
    def open(cls, dataset_path: Union[str, PathLike]) -> "Dataset":
        """
        To open an existing dataset on disk, simply call `Dataset.open("your_path")`.
        This requires `datasource-properties.json` to exist in this folder. Based on the `datasource-properties.json`,
        a dataset object is constructed. Only layers and magnifications that are listed in the properties are loaded
        (even though there might exist more layers or magnifications on disk).

        The `dataset_path` refers to the top level directory of the dataset (excluding layer or magnification names).
        """
        dataset_path = UPath(dataset_path)
        assert (
            dataset_path.exists()
        ), f"Cannot open Dataset: Couldn't find {dataset_path}"
        assert (
            dataset_path.is_dir()
        ), f"Cannot open Dataset: {dataset_path} is not a directory"
        assert (
            dataset_path / PROPERTIES_FILE_NAME
        ).is_file(), (
            f"Cannot open Dataset: Could not find {dataset_path / PROPERTIES_FILE_NAME}"
        )

        return cls(dataset_path, voxel_size=_UNSPECIFIED_SCALE_FROM_OPEN, exist_ok=True)

    @classmethod
    def _parse_remote(
        cls,
        dataset_name_or_url: str,
        organization_id: Optional[str] = None,
        sharing_token: Optional[str] = None,
        webknossos_url: Optional[str] = None,
    ) -> Tuple[ContextManager, str, str, Optional[str]]:
        """Parses the given arguments to
        * context_manager that should be entered,
        * dataset_name,
        * organization_id,
        * sharing_token.
        """
        from webknossos.client.context import _get_context, webknossos_context

        caller = inspect.stack()[1].function

        match = re.match(_DATASET_URL_REGEX, dataset_name_or_url)
        if match is not None:
            assert (
                organization_id is None
                and sharing_token is None
                and webknossos_url is None
            ), (
                f"When Dataset.{caller}() is called with an annotation url, "
                + f"e.g. Dataset.{caller}('https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view'), "
                + "organization_id, sharing_token and webknossos_url must not be set."
            )
            dataset_name = match.group("dataset_name")
            organization_id = match.group("organization_id")
            sharing_token = match.group("sharing_token")
            webknossos_url = match.group("webknossos_url")
        else:
            dataset_name = dataset_name_or_url

        current_context = _get_context()
        if webknossos_url is not None and webknossos_url != current_context.url:
            if sharing_token is None:
                warnings.warn(
                    f"The supplied url {webknossos_url} does not match your current context {current_context.url}. "
                    + f"Using no token, only public datasets can used with Dataset.{caller}(). "
                    + "Please see https://docs.webknossos.org/api/webknossos/client/context.html to adapt the URL and token."
                )
            assert organization_id is not None, (
                f"Please supply the organization_id to Dataset.{caller}()."
                f"The supplied url {webknossos_url} does not match your current context {current_context.url}. "
                + "In this case organization_id can not be inferred."
            )
            context_manager: ContextManager[None] = webknossos_context(
                webknossos_url, token=None
            )
        else:
            if organization_id is None:
                organization_id = current_context.organization_id

            context_manager = nullcontext()

        return context_manager, dataset_name, organization_id, sharing_token

    @classmethod
    def open_remote(
        cls,
        dataset_name_or_url: str,
        organization_id: Optional[str] = None,
        sharing_token: Optional[str] = None,
        webknossos_url: Optional[str] = None,
    ) -> "RemoteDataset":
        """Opens a remote webknossos dataset. Image data is accessed via network requests.
        Dataset metadata such as allowed teams or the sharing token can be read and set
        via the respective `RemoteDataset` properties.

        * `dataset_name_or_url` may be a dataset name or a full URL to a dataset view, e.g.
          `https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view`
          If a URL is used, `organization_id`, `webknossos_url` and `sharing_token` must not be set.
        * `organization_id` may be supplied if a dataset name was used in the previous argument,
          it defaults to your current organization from the `webknossos_context`.
          You can find your `organization_id` [here](https://webknossos.org/auth/token).
        * `sharing_token` may be supplied if a dataset name was used and can specify a sharing token.
        * `webknossos_url` may be supplied if a dataset name was used,
          and allows to specifiy in which webknossos instance to search for the dataset.
          It defaults to the url from your current `webknossos_context`, using https://webknossos.org as a fallback.
        """
        from webknossos.client._generated.api.default import dataset_info
        from webknossos.client.context import _get_context

        (
            context_manager,
            dataset_name,
            organization_id,
            sharing_token,
        ) = cls._parse_remote(
            dataset_name_or_url, organization_id, sharing_token, webknossos_url
        )

        with context_manager:
            wk_context = _get_context()
            dataset_info_response = dataset_info.sync_detailed(
                organization_name=organization_id,
                data_set_name=dataset_name,
                client=wk_context.generated_client,
                sharing_token=sharing_token,
            )
            assert dataset_info_response.status_code == 200, dataset_info_response
            parsed = dataset_info_response.parsed
            assert parsed is not None

            token = sharing_token or wk_context.datastore_token

        datastore_url = parsed.data_store.url

        zarr_path = UPath(
            f"{datastore_url}/data/zarr/{organization_id}/{dataset_name}/",
            headers={} if token is None else {"X-Auth-Token": token},
        )
        return RemoteDataset(
            zarr_path, dataset_name, organization_id, sharing_token, context_manager
        )

    @classmethod
    def download(
        cls,
        dataset_name_or_url: str,
        organization_id: Optional[str] = None,
        sharing_token: Optional[str] = None,
        webknossos_url: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        layers: Union[List[str], str, None] = None,
        mags: Optional[List[Mag]] = None,
        path: Optional[Union[PathLike, str]] = None,
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
          and allows to specifiy in which webknossos instance to search for the dataset.
          It defaults to the url from your current `webknossos_context`, using https://webknossos.org as a fallback.
        * `bbox`, `layers`, and `mags` specify which parts of the dataset to download.
          If nothing is specified the whole image, all layers, and all mags are downloaded respectively.
        * `path` and `exist_ok` specify where to save the downloaded dataset and whether to overwrite
          if the `path` exists.
        """

        from webknossos.client._download_dataset import download_dataset

        (
            context_manager,
            dataset_name,
            organization_id,
            sharing_token,
        ) = cls._parse_remote(
            dataset_name_or_url, organization_id, sharing_token, webknossos_url
        )

        if isinstance(layers, str):
            layers = [layers]

        with context_manager:
            return download_dataset(
                dataset_name,
                organization_id=organization_id,
                sharing_token=sharing_token,
                bbox=bbox,
                layers=layers,
                mags=mags,
                path=path,
                exist_ok=exist_ok,
            )

    @classmethod
    def from_images(
        cls,
        input_path: Union[str, PathLike],
        output_path: Union[str, PathLike],
        voxel_size: Tuple[float, float, float],
        name: Optional[str] = None,
        *,
        map_filepath_to_layer_name: Union[
            ConversionLayerMapping, Callable[[Path], str]
        ] = ConversionLayerMapping.INSPECT_SINGLE_FILE,
        z_slices_sort_key: Callable[[Path], Any] = natsort_keygen(),
        layer_category: Optional[LayerCategoryType] = None,
        data_format: Union[str, DataFormat] = DEFAULT_DATA_FORMAT,
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[int, Vec3IntLike]] = None,
        compress: bool = False,
        swap_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        use_bioformats: bool = False,
        max_layers: int = 20,
        batch_size: Optional[int] = None,
        executor: Optional[Executor] = None,
    ) -> "Dataset":
        """
        This method imports image data in a folder as a webKnossos dataset. The
        image data can be 3D images (such as multipage tiffs) or stacks of 2D
        images. In case of multiple 3D images or image stacks, those are mapped
        to different layers. The exact mapping is handled by the argument
        `map_filepath_to_layer_name`, which can be a pre-defined strategy from
        the enum `ConversionLayerMapping`, or a custom callable, taking
        a path of an image file and returning the corresponding layer name. All
        files belonging to the same layer name are then grouped. In case of
        multiple files per layer, those are usually mapped to the z-dimension.
        The order of the z-slices can be customized by setting
        `z_slices_sort_key`.

        The category of layers (`color` vs `segmentation`) is determined
        automatically by checking if `segmentation` is part of the path.
        Alternatively, a category can be enforced by passing `layer_category`.

        Further arguments behave as in `add_layer_from_images`, please also
        refer to its documentation.

        For more fine-grained control, please create an empty dataset and use
        `add_layer_from_images`.
        """
        from ._utils.pims_images import get_valid_pims_suffixes

        input_upath = UPath(input_path)

        valid_suffixes = get_valid_pims_suffixes()

        input_files = [
            i.relative_to(input_upath)
            for i in input_upath.glob("**/*")
            if i.is_file() and i.suffix.lstrip(".") in valid_suffixes
        ]
        if len(input_files) == 0:
            raise ValueError(
                "Could not find any supported image data. "
                + f"The following suffixes are supported: {sorted(valid_suffixes)}"
            )

        if isinstance(map_filepath_to_layer_name, Dataset.ConversionLayerMapping):
            map_filepath_to_layer_name = map_filepath_to_layer_name._to_callable(
                input_upath, input_files=input_files, use_bioformats=use_bioformats
            )

        ds = cls(output_path, voxel_size=voxel_size, name=name)

        filepaths_per_layer: Dict[str, List[Path]] = {}
        for input_file in input_files:
            layer_name = map_filepath_to_layer_name(input_file)
            filepaths_per_layer.setdefault(layer_name, []).append(
                input_path / input_file
            )

        for layer_name, filepaths in filepaths_per_layer.items():
            filepaths.sort(key=z_slices_sort_key)
            category: LayerCategoryType
            if layer_category is None:
                category = (
                    "segmentation"
                    if guess_if_segmentation_path(filepaths[0])
                    else "color"
                )
            else:
                category = layer_category
            ds.add_layer_from_images(
                filepaths[0] if len(filepaths) == 1 else filepaths,
                layer_name,
                category=category,
                data_format=data_format,
                chunk_shape=chunk_shape,
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
                executor=executor,
            )

        return ds

    @property
    def layers(self) -> Dict[str, Layer]:
        """
        Getter for dictionary containing all layers.
        """
        return self._layers

    @property
    def voxel_size(self) -> Tuple[float, float, float]:
        return self._properties.scale

    @property
    def scale(self) -> Tuple[float, float, float]:
        """Deprecated, use `voxel_size` instead."""
        warn_deprecated("scale", "voxel_size")
        return self._properties.scale

    @property
    def name(self) -> str:
        return self._properties.id["name"]

    @name.setter
    def name(self, name: str) -> None:
        self._ensure_writable()
        current_id = self._properties.id
        current_id["name"] = name
        self._properties.id = current_id
        self._export_as_json()

    @property
    def default_view_configuration(self) -> Optional[DatasetViewConfiguration]:
        return self._properties.default_view_configuration

    @default_view_configuration.setter
    def default_view_configuration(
        self, view_configuration: DatasetViewConfiguration
    ) -> None:
        self._ensure_writable()
        self._properties.default_view_configuration = view_configuration
        self._export_as_json()  # update properties on disk

    @property
    def read_only(self) -> bool:
        return self._read_only

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.path == other.path and self.read_only == other.read_only
        else:
            return False

    def upload(
        self,
        new_dataset_name: Optional[str] = None,
        layers_to_link: Optional[List[Union["LayerToLink", Layer]]] = None,
        jobs: Optional[int] = None,
    ) -> "RemoteDataset":
        """
        Uploads this dataset to webKnossos.

        The `new_dataset_name` parameter allows to assign a specific name for the dataset.
        `layers_to_link` allows to add (or override) a layer in the uploaded dataset, so that
        it links to a layer of an existing dataset in webKnossos. That way, already existing
        layers don't need to be uploaded again.

        If supplied, the `jobs` parameter will determine the number of simultaneous chunk uploads. Defaults to 5.

        Returns the `RemoteDataset` upon successful upload.
        """

        from webknossos.client._upload_dataset import LayerToLink, upload_dataset

        converted_layers_to_link = (
            None
            if layers_to_link is None
            else [
                i if isinstance(i, LayerToLink) else LayerToLink.from_remote_layer(i)
                for i in layers_to_link
            ]
        )

        return self.open_remote(
            upload_dataset(self, new_dataset_name, converted_layers_to_link, jobs)
        )

    def get_layer(self, layer_name: str) -> Layer:
        """
        Returns the layer called `layer_name` of this dataset. The return type is `webknossos.dataset.layer.Layer`.

        This function raises an `IndexError` if the specified `layer_name` does not exist.
        """
        if layer_name not in self.layers.keys():
            raise IndexError(
                "The layer {} is not a layer of this dataset".format(layer_name)
            )
        return self.layers[layer_name]

    def add_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        dtype_per_layer: Optional[DTypeLike] = None,
        dtype_per_channel: Optional[DTypeLike] = None,
        num_channels: Optional[int] = None,
        data_format: Union[str, DataFormat] = DEFAULT_DATA_FORMAT,
        **kwargs: Any,
    ) -> Layer:
        """
        Creates a new layer called `layer_name` and adds it to the dataset.
        The dtype can either be specified per layer or per channel.
        If neither of them are specified, `uint8` per channel is used as default.
        When creating a "Segmentation Layer" (`category="segmentation"`),
        the parameter `largest_segment_id` also has to be specified.

        Creates the folder `layer_name` in the directory of `self.path`.

        WKW layers can only be added to datasets on local file systems.

        The return type is `webknossos.dataset.layer.Layer`.

        This function raises an `IndexError` if the specified `layer_name` already exists.
        """

        self._ensure_writable()

        if "dtype" in kwargs:
            raise ValueError(
                f"Called Dataset.add_layer with 'dtype'={kwargs['dtype']}. This parameter is deprecated. Use 'dtype_per_layer' or 'dtype_per_channel' instead."
            )
        if num_channels is None:
            num_channels = 1

        if dtype_per_layer is not None and dtype_per_channel is not None:
            raise AttributeError(
                "Cannot add layer. Specifying both 'dtype_per_layer' and 'dtype_per_channel' is not allowed"
            )
        elif dtype_per_channel is not None:
            dtype_per_channel = _properties_floating_type_to_python_type.get(
                dtype_per_channel, dtype_per_channel  # type: ignore[arg-type]
            )
            dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)  # type: ignore[arg-type]
        elif dtype_per_layer is not None:
            dtype_per_layer = _properties_floating_type_to_python_type.get(
                dtype_per_layer, dtype_per_layer  # type: ignore[arg-type]
            )
            dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)  # type: ignore[arg-type]
            dtype_per_channel = _dtype_per_layer_to_dtype_per_channel(
                dtype_per_layer, num_channels
            )
        else:
            dtype_per_channel = np.dtype("uint" + str(DEFAULT_BIT_DEPTH))

        if layer_name in self.layers.keys():
            raise IndexError(
                "Adding layer {} failed. There is already a layer with this name".format(
                    layer_name
                )
            )

        assert (
            is_fs_path(self.path) or data_format != DataFormat.WKW
        ), "Cannot create WKW layers in remote datasets. Use `data_format='zarr'`."

        layer_properties = LayerProperties(
            name=layer_name,
            category=category,
            bounding_box=BoundingBox((0, 0, 0), (0, 0, 0)),
            element_class=_dtype_per_channel_to_element_class(
                dtype_per_channel, num_channels
            ),
            mags=[],
            num_channels=num_channels,
            data_format=DataFormat(data_format),
        )

        if category == COLOR_CATEGORY:
            self._properties.data_layers += [layer_properties]
            self._layers[layer_name] = Layer(self, layer_properties)
        elif category == SEGMENTATION_CATEGORY:
            assert (
                "largest_segment_id" in kwargs
            ), f"Failed to create segmentation layer {layer_name}: the parameter 'largest_segment_id' was not specified, which is necessary for segmentation layers."

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
                self, segmentation_layer_properties
            )
        else:
            raise RuntimeError(
                f"Failed to add layer ({layer_name}) because of invalid category ({category}). The supported categories are '{COLOR_CATEGORY}' and '{SEGMENTATION_CATEGORY}'"
            )

        self._export_as_json()
        return self.layers[layer_name]

    def get_or_add_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        dtype_per_layer: Optional[DTypeLike] = None,
        dtype_per_channel: Optional[DTypeLike] = None,
        num_channels: Optional[int] = None,
        data_format: Union[str, DataFormat] = DEFAULT_DATA_FORMAT,
        **kwargs: Any,
    ) -> Layer:
        """
        Creates a new layer called `layer_name` and adds it to the dataset, in case it did not exist before.
        Then, returns the layer.

        For more information see `add_layer`.
        """

        if "dtype" in kwargs:
            raise ValueError(
                f"Called Dataset.get_or_add_layer with 'dtype'={kwargs['dtype']}. This parameter is deprecated. Use 'dtype_per_layer' or 'dtype_per_channel' instead."
            )
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

    def add_layer_like(self, other_layer: Layer, layer_name: str) -> Layer:
        self._ensure_writable()

        if layer_name in self.layers.keys():
            raise IndexError(
                f"Adding layer {layer_name} failed. There is already a layer with this name"
            )

        layer_properties = copy.copy(other_layer._properties)
        layer_properties.mags = []
        layer_properties.name = layer_name

        self._properties.data_layers += [layer_properties]
        if layer_properties.category == COLOR_CATEGORY:
            self._layers[layer_name] = Layer(self, layer_properties)
        elif layer_properties.category == SEGMENTATION_CATEGORY:
            self._layers[layer_name] = SegmentationLayer(self, layer_properties)
        else:
            raise RuntimeError(
                f"Failed to add layer ({layer_name}) because of invalid category ({layer_properties.category}). The supported categories are '{COLOR_CATEGORY}' and '{SEGMENTATION_CATEGORY}'"
            )
        self._export_as_json()
        return self._layers[layer_name]

    def add_layer_for_existing_files(
        self,
        layer_name: str,
        category: LayerCategoryType,
        **kwargs: Any,
    ) -> Layer:
        self._ensure_writable()
        assert layer_name not in self.layers, f"Layer {layer_name} already exists!"

        array_info = _find_array_info(self.path / layer_name)
        assert (
            array_info is not None
        ), f"Could not find any valid mags in {self.path /layer_name}. Cannot add layer."
        layer = self.add_layer(
            layer_name,
            category=category,
            num_channels=array_info.num_channels,
            dtype_per_channel=array_info.voxel_type,
            data_format=array_info.data_format,
            **kwargs,
        )
        for mag_dir in layer.path.iterdir():
            layer.add_mag_for_existing_files(mag_dir.name)
        finest_mag_view = layer.mags[min(layer.mags)]
        layer.bounding_box = infer_bounding_box_existing_files(finest_mag_view)
        return layer

    def add_layer_from_images(
        self,
        images: Union[str, "pims.FramesSequence", List[Union[str, PathLike]]],
        ## add_layer arguments
        layer_name: str,
        category: LayerCategoryType = "color",
        data_format: Union[str, DataFormat] = DEFAULT_DATA_FORMAT,
        ## add_mag arguments
        mag: Union[int, str, list, tuple, np.ndarray, Mag] = Mag(1),
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[int, Vec3IntLike]] = None,
        compress: bool = False,
        *,
        ## other arguments
        topleft: Vec3IntLike = Vec3Int.zeros(),  # in Mag(1)
        swap_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        dtype: Optional[DTypeLike] = None,
        use_bioformats: bool = False,
        channel: Optional[int] = None,
        timepoint: Optional[int] = None,
        czi_channel: Optional[int] = None,
        batch_size: Optional[int] = None,  # defaults to shard-size z
        allow_multiple_layers: bool = False,
        max_layers: int = 20,
        truncate_rgba_to_rgb: bool = True,
        executor: Optional[Executor] = None,
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,  # deprecated
    ) -> Layer:
        """
        Creates a new layer called `layer_name` with mag `mag` from `images`.
        `images` can be one of the following:
        * glob-string
        * list of paths
        * `pims.FramesSequence` instance

        Please see the [pims docs](http://soft-matter.github.io/pims/v0.6.1/opening_files.html) for more information.

        This method needs extra packages such as pims. Please install the respective extras,
        e.g. using `python -m pip install "webknossos[all]"`.

        Further Arguments:
        * `category`: `color` by default, may be set to "segmentation"
        * `data_format`: by default wkw files are written, may be set to "zarr"
        * `mag`: magnification to use for the written data
        * `chunk_shape`, `chunks_per_shard`, `compress`: adjust how the data is stored on disk
        * `topleft`: set an offset in Mag(1) to start writing the data, only affecting the output
        * `swap_xy`: set to `True` to interchange x and y axis before writing to disk
        * `flip_x`, `flip_y`, `flip_z`: set to `True` to reverse the respective axis before writing to disk
        * `dtype`: the read image data will be convertoed to this dtype using `numpy.ndarray.astype`
        * `use_bioformats`: set to `True` to only use the
          [pims bioformats adapter](https://soft-matter.github.io/pims/v0.6.1/bioformats.html) directly, needs a JVM
        * `channel`: may be used to select a single channel, if multiple are available
        * `timepoint`: for timeseries, select a timepoint to use by specifying it as an int, starting from 0
        * `czi_channel`: may be used to select a channel for .czi images, which differs from normal color-channels
        * `batch_size`: size to process the images, must be a multiple of the chunk-size z-axis for uncompressed and the shard-size z-axis for compressed layers, default is the chunk-size or shard-size respectively
        * `allow_multiple_layers`: set to `True` if timepoints or channels may result in multiple layers being added (only the first is returned)
        * `max_layers`: only applies if `allow_multiple_layers=True`, limits the number of layers added via different channels or timepoints
        * `truncate_rgba_to_rgb`: only applies if `allow_multiple_layers=True`, set to `False` to write four channels into layers instead of an RGB channel
        * `executor`: pass a `ClusterExecutor` instance to parallelize the conversion jobs across the batches
        """
        from ._utils.pims_images import PimsImages, dimwise_max

        chunk_shape, chunks_per_shard = _get_sharding_parameters(
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            chunk_size=chunk_size,
            block_len=None,
            file_len=None,
        )

        pims_images = PimsImages(
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
        possible_layers = pims_images.get_possible_layers()
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
                #    "__channel=0_timepoint=0", {"channel": 0, "timepoint": 0},
                #    "__channel=0_timepoint=1", {"channel": 0, "timepoint": 1},
                #    "__channel=0_timepoint=2", {"channel": 0, "timepoint": 2},
                #    …,
                #    "__channel=1_timepoint=0", {"channel": 1, "timepoint": 0},
                #    …,
                # }
                suffix_with_pims_open_kwargs_per_layer = {
                    "__" + "_".join(f"{k}={v}" for k, v in sorted(pairs)): dict(pairs)
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
                    f"There are dimensions beyond channels and xyz which cannot be read: {possible_layers}. "
                    "Defaulting to the first one. "
                    "Please set allow_multiple_layers=True if all of them should be written to different layers, "
                    "or set specific values as arguments.",
                    RuntimeWarning,
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
                f"Limiting the number of added layers to {max_layers} out of {len(suffix_with_pims_open_kwargs_per_layer)}. "
                + "Please increase max_layers if you want more layers to be added.",
                RuntimeWarning,
            )
        for _, (
            layer_name_suffix,
            pims_open_kwargs,
        ) in zip(range(max_layers), suffix_with_pims_open_kwargs_per_layer.items()):
            # If pims_open_kwargs is empty there's no need to re-open the images:
            if len(pims_open_kwargs) > 0:
                # Set parameters from this method as default
                # if they are not part of the kwargs per layer:
                pims_open_kwargs.setdefault("timepoint", timepoint)
                pims_open_kwargs.setdefault("channel", channel)
                pims_open_kwargs.setdefault("czi_channel", czi_channel)
                pims_images = PimsImages(
                    images,
                    swap_xy=swap_xy,
                    flip_x=flip_x,
                    flip_y=flip_y,
                    flip_z=flip_z,
                    use_bioformats=use_bioformats,
                    is_segmentation=category == "segmentation",
                    **pims_open_kwargs,
                )
            layer = self.add_layer(
                layer_name=layer_name + layer_name_suffix,
                category=category,
                data_format=data_format,
                dtype_per_channel=pims_images.dtype if dtype is None else dtype,
                num_channels=pims_images.num_channels,
                **add_layer_kwargs,  # type: ignore[arg-type]
            )
            mag_view = layer.add_mag(
                mag=mag,
                chunk_shape=chunk_shape,
                chunks_per_shard=chunks_per_shard,
                compress=compress,
            )
            mag = mag_view.mag
            layer.bounding_box = (
                BoundingBox((0, 0, 0), pims_images.expected_shape)
                .from_mag_to_mag1(mag)
                .offset(topleft)
            )

            if batch_size is None:
                if compress:
                    batch_size = mag_view.info.shard_shape.z
                else:
                    batch_size = mag_view.info.chunk_shape.z
            elif compress:
                assert (
                    batch_size % mag_view.info.shard_shape.z == 0
                ), f"batch_size {batch_size} must be divisible by z shard-size {mag_view.info.shard_shape.z} when creating compressed layers"
            else:
                assert (
                    batch_size % mag_view.info.chunk_shape.z == 0
                ), f"batch_size {batch_size} must be divisible by z chunk-size {mag_view.info.chunk_shape.z}"

            func_per_chunk = named_partial(
                pims_images.copy_to_view,
                mag_view=mag_view,
                is_segmentation=category == "segmentation",
                dtype=dtype,
            )

            args = []
            for z_start in range(0, pims_images.expected_shape.z, batch_size):
                z_end = min(z_start + batch_size, pims_images.expected_shape.z)
                # return shapes and set to union when using --pad
                args.append((z_start, z_end))
            with warnings.catch_warnings():
                # Block alignmnent within the dataset should not be a problem, since shard-wise chunking is enforced.
                # However, dataset borders might change between different parallelized writes, when sizes differ.
                # For differing sizes, a separate warning is thrown, so block alignment warnings can be ignored:
                warnings.filterwarnings(
                    "ignore",
                    message=_BLOCK_ALIGNMENT_WARNING,
                    category=RuntimeWarning,
                    module="webknossos",
                )
                # There are race-conditions about setting the bbox of the layer.
                # The bbox is set correctly afterwards, ignore errors here:
                warnings.filterwarnings(
                    "ignore",
                    message=".*properties were found on disk which are newer than the ones that were seen last time.*",
                    category=UserWarning,
                    module="webknossos",
                )
                with get_executor_for_args(None, executor) as executor:
                    shapes_and_max_ids = wait_and_ensure_success(
                        executor.map_to_futures(func_per_chunk, args),
                        progress_desc="Creating layer from images",
                    )
                shapes, max_ids = zip(*shapes_and_max_ids)
                if category == "segmentation":
                    max_id = max(max_ids)
                    cast(SegmentationLayer, layer).largest_segment_id = max_id
                actual_size = Vec3Int(
                    dimwise_max(shapes) + (pims_images.expected_shape.z,)
                )
                layer.bounding_box = (
                    BoundingBox((0, 0, 0), actual_size)
                    .from_mag_to_mag1(mag)
                    .offset(topleft)
                )
            if pims_images.expected_shape != actual_size:
                warnings.warn(
                    "Some images are larger than expected, smaller slices are padded with zeros now. "
                    + f"New size is {actual_size}, expected {pims_images.expected_shape}.",
                    RuntimeWarning,
                )
            if first_layer is None:
                first_layer = layer
        assert first_layer is not None
        return first_layer

    def get_segmentation_layer(self) -> SegmentationLayer:
        """
        Deprecated, please use `get_segmentation_layers()`.

        Returns the only segmentation layer.
        Fails with a IndexError if there are multiple segmentation layers or none.
        """

        warnings.warn(
            "[DEPRECATION] get_segmentation_layer() fails if no or more than one segmentation layer exists. Prefer get_segmentation_layers().",
            DeprecationWarning,
        )
        return cast(
            SegmentationLayer,
            self._get_layer_by_category(SEGMENTATION_CATEGORY),
        )

    def get_segmentation_layers(self) -> List[SegmentationLayer]:
        """
        Returns all segmentation layers.
        """
        return [
            cast(SegmentationLayer, layer)
            for layer in self.layers.values()
            if layer.category == SEGMENTATION_CATEGORY
        ]

    def get_color_layer(self) -> Layer:
        """
        Deprecated, please use `get_color_layers()`.

        Returns the only color layer.
        Fails with a RuntimeError if there are multiple color layers or none.
        """
        warnings.warn(
            "[DEPRECATION] get_color_layer() fails if no or more than one color layer exists. Prefer get_color_layers().",
            DeprecationWarning,
        )
        return self._get_layer_by_category(COLOR_CATEGORY)

    def get_color_layers(self) -> List[Layer]:
        """
        Returns all color layers.
        """
        return [
            cast(Layer, layer)
            for layer in self.layers.values()
            if layer.category == COLOR_CATEGORY
        ]

    def delete_layer(self, layer_name: str) -> None:
        """
        Deletes the layer from the `datasource-properties.json` and the data from disk.
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
        self._export_as_json()

    def add_copy_layer(
        self,
        foreign_layer: Union[str, Path, Layer],
        new_layer_name: Optional[str] = None,
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        data_format: Optional[Union[str, DataFormat]] = None,
        compress: Optional[bool] = None,
        executor: Optional[Executor] = None,
    ) -> Layer:
        """
        Copies the data at `foreign_layer` which belongs to another dataset to the current dataset.
        Additionally, the relevant information from the `datasource-properties.json` of the other dataset are copied too.
        If new_layer_name is None, the name of the foreign layer is used.
        """
        self._ensure_writable()
        foreign_layer = Layer._ensure_layer(foreign_layer)

        if new_layer_name is None:
            new_layer_name = foreign_layer.name

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
        )
        layer.bounding_box = foreign_layer.bounding_box

        for mag_view in foreign_layer.mags.values():
            layer.add_copy_mag(
                mag_view,
                extend_layer_bounding_box=False,
                chunk_shape=chunk_shape,
                chunks_per_shard=chunks_per_shard,
                compress=compress,
                executor=executor,
            )

        return layer

    def add_symlink_layer(
        self,
        foreign_layer: Union[str, Path, Layer],
        make_relative: bool = False,
        new_layer_name: Optional[str] = None,
    ) -> Layer:
        """
        Creates a symlink to the data at `foreign_layer` which belongs to another dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        If make_relative is True, the symlink is made relative to the current dataset path.
        If new_layer_name is None, the name of the foreign layer is used.
        Symlinked layers can only be added to datasets on local file systems.
        """
        self._ensure_writable()
        foreign_layer = Layer._ensure_layer(foreign_layer)

        if new_layer_name is None:
            new_layer_name = foreign_layer.name

        if new_layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer}. This dataset already has a layer called {new_layer_name}."
            )
        foreign_layer_path = foreign_layer.path

        assert is_fs_path(
            self.path
        ), f"Cannot create symlinks in remote dataset {self.path}"
        assert is_fs_path(
            foreign_layer_path
        ), f"Cannot create symlink to remote layer {foreign_layer_path}"

        foreign_layer_symlink_path = (
            Path(relpath(foreign_layer_path, self.path))
            if make_relative
            else foreign_layer_path.resolve()
        )

        (self.path / new_layer_name).symlink_to(foreign_layer_symlink_path)
        layer_properties = copy.deepcopy(foreign_layer._properties)
        layer_properties.name = new_layer_name
        self._properties.data_layers += [layer_properties]
        self._layers[new_layer_name] = self._initialize_layer_from_properties(
            layer_properties
        )

        self._export_as_json()
        return self.layers[new_layer_name]

    def add_fs_copy_layer(
        self,
        foreign_layer: Union[str, Path, Layer],
        new_layer_name: Optional[str] = None,
    ) -> Layer:
        """
        Copies the files at `foreign_layer` which belongs to another dataset
        to the current dataset via the filesystem. Additionally, the relevant
        information from the `datasource-properties.json` of the other dataset
        are copied too. If new_layer_name is None, the name of the foreign
        layer is used.
        """
        self._ensure_writable()
        foreign_layer = Layer._ensure_layer(foreign_layer)

        if new_layer_name is None:
            new_layer_name = foreign_layer.name

        if new_layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot copy {foreign_layer}. This dataset already has a layer called {new_layer_name}."
            )

        copytree(foreign_layer.path, self.path / new_layer_name)
        layer_properties = copy.deepcopy(foreign_layer._properties)
        layer_properties.name = new_layer_name
        self._properties.data_layers += [layer_properties]
        self._layers[new_layer_name] = self._initialize_layer_from_properties(
            layer_properties
        )

        self._export_as_json()
        return self.layers[new_layer_name]

    def copy_dataset(
        self,
        new_dataset_path: Union[str, Path],
        voxel_size: Optional[Tuple[float, float, float]] = None,
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        data_format: Optional[Union[str, DataFormat]] = None,
        compress: Optional[bool] = None,
        args: Optional[Namespace] = None,  # deprecated
        executor: Optional[Executor] = None,
        *,
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,  # deprecated
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
    ) -> "Dataset":
        """
        Creates a new dataset at `new_dataset_path` and copies the data from the current dataset to `empty_target_ds`.
        If not specified otherwise, the `voxel_size`, `chunk_shape`, `chunks_per_shard` and `compress` of the current dataset
        are also used for the new dataset.
        WKW layers can only be copied to datasets on local file systems.
        """

        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        chunk_shape, chunks_per_shard = _get_sharding_parameters(
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            chunk_size=chunk_size,
            block_len=block_len,
            file_len=file_len,
        )

        new_dataset_path = UPath(new_dataset_path)

        if data_format == DataFormat.WKW:
            assert is_fs_path(
                new_dataset_path
            ), "Cannot create WKW-based remote datasets. Use `data_format='zarr'` instead."
        if data_format is None and any(
            layer.data_format == DataFormat.WKW for layer in self.layers.values()
        ):
            assert is_fs_path(
                new_dataset_path
            ), "Cannot create WKW layers in remote datasets. Use explicit `data_format='zarr'`."

        if voxel_size is None:
            voxel_size = self.voxel_size
        new_ds = Dataset(new_dataset_path, voxel_size=voxel_size, exist_ok=False)

        with get_executor_for_args(args, executor) as executor:
            for layer in self.layers.values():
                new_ds.add_copy_layer(
                    layer,
                    chunk_shape=chunk_shape,
                    chunks_per_shard=chunks_per_shard,
                    data_format=data_format,
                    compress=compress,
                    executor=executor,
                )
        new_ds._export_as_json()
        return new_ds

    def shallow_copy_dataset(
        self,
        new_dataset_path: Union[str, PathLike],
        name: Optional[str] = None,
        make_relative: bool = False,
        layers_to_ignore: Optional[Iterable[str]] = None,
    ) -> "Dataset":
        """
        Create a new dataset at the given path. Link all mags of all existing layers.
        In addition, link all other directories in all layer directories
        to make this method robust against additional files e.g. layer/mappings/agglomerate_view.hdf5.
        This method becomes useful when exposing a dataset to webknossos.
        Only datasets on local filesystems can be shallow copied.
        """
        assert is_fs_path(
            self.path
        ), f"Cannot create symlinks to remote dataset {self.path}"
        new_dataset_path = UPath(new_dataset_path)
        assert is_fs_path(
            new_dataset_path
        ), f"Cannot create symlink in remote path {new_dataset_path}"
        new_dataset = Dataset(
            new_dataset_path,
            voxel_size=self.voxel_size,
            name=name or self.name,
            exist_ok=False,
        )
        for layer_name, layer in self.layers.items():
            if layers_to_ignore is not None and layer_name in layers_to_ignore:
                continue
            new_layer = new_dataset.add_layer_like(layer, layer_name)
            for mag_view in layer.mags.values():
                new_layer.add_symlink_mag(mag_view, make_relative)

            # copy all other directories with a dir scan
            copy_directory_with_symlinks(
                layer.path,
                new_layer.path,
                ignore=[str(mag) for mag in layer.mags]
                + [PROPERTIES_FILE_NAME, ZGROUP_FILE_NAME, ZATTRS_FILE_NAME],
                make_relative=make_relative,
            )

        return new_dataset

    def compress(
        self,
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Compresses all mag views in-place that are not yet compressed.
        """
        for layer in self.layers.values():
            for mag in layer.mags.values():
                if not mag._is_compressed():
                    mag.compress(executor=executor)

    def downsample(
        self,
        sampling_mode: SamplingModes = SamplingModes.ANISOTROPIC,
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Downsamples all layers that are not yet downsampled.
        """
        for layer in self.layers.values():
            layer.downsample(
                sampling_mode=sampling_mode,
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

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, PathLike],
        voxel_size: Tuple[float, float, float],
        name: Optional[str] = None,
    ) -> "Dataset":
        """
        **Deprecated**, please use the constructor `Dataset()` instead.
        """
        warnings.warn(
            "[DEPRECATION] Dataset.create() is deprecated in favor of the normal constructor Dataset().",
            DeprecationWarning,
        )
        return cls(dataset_path, voxel_size, name, exist_ok=False)

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        voxel_size: Tuple[float, float, float],
        name: Optional[str] = None,
    ) -> "Dataset":
        """
        **Deprecated**, please use the constructor `Dataset()` instead.
        """
        warnings.warn(
            "[DEPRECATION] Dataset.get_or_create() is deprecated in favor of the normal constructor Dataset(…, exist_ok=True).",
            DeprecationWarning,
        )
        return cls(dataset_path, voxel_size, name, exist_ok=True)

    def __repr__(self) -> str:
        return f"Dataset({repr(self.path)})"

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise RuntimeError(f"{self} is read-only, the changes will not be saved!")

    def _load_properties(self) -> DatasetProperties:
        with (self.path / PROPERTIES_FILE_NAME).open(
            encoding="utf-8"
        ) as datasource_properties:
            data = json.load(datasource_properties)
        return dataset_converter.structure(data, DatasetProperties)

    def _export_as_json(self) -> None:
        self._ensure_writable()

        properties_on_disk = self._load_properties()

        if properties_on_disk != self._last_read_properties:
            warnings.warn(
                "[WARNING] While exporting the dataset's properties, properties were found on disk which are "
                + "newer than the ones that were seen last time. The properties will be overwritten. This is "
                + "likely happening because multiple processes changed the metadata of this dataset."
            )

        with (self.path / PROPERTIES_FILE_NAME).open("w", encoding="utf-8") as outfile:
            json.dump(
                dataset_converter.unstructure(self._properties),
                outfile,
                indent=4,
            )

            self._last_read_properties = copy.deepcopy(self._properties)

        # Write out Zarr and OME-Ngff metadata if there is a Zarr layer
        if any(layer.data_format == DataFormat.Zarr for layer in self.layers.values()):
            zgroup_content = {"zarr_format": "2"}
            with (self.path / ZGROUP_FILE_NAME).open("w", encoding="utf-8") as outfile:
                json.dump(zgroup_content, outfile, indent=4)
            for layer in self.layers.values():
                if layer.data_format == DataFormat.Zarr:
                    with (layer.path / ZGROUP_FILE_NAME).open(
                        "w", encoding="utf-8"
                    ) as outfile:
                        json.dump(zgroup_content, outfile, indent=4)
                    with (layer.path / ZATTRS_FILE_NAME).open(
                        "w", encoding="utf-8"
                    ) as outfile:
                        json.dump(
                            {
                                "multiscales": [
                                    {
                                        "version": "0.4",
                                        "axes": [
                                            {"name": "c", "type": "channel"},
                                            {
                                                "name": "x",
                                                "type": "space",
                                                "unit": "nanometer",
                                            },
                                            {
                                                "name": "y",
                                                "type": "space",
                                                "unit": "nanometer",
                                            },
                                            {
                                                "name": "z",
                                                "type": "space",
                                                "unit": "nanometer",
                                            },
                                        ],
                                        "datasets": [
                                            {
                                                "path": mag.path.name,
                                                "coordinateTransformations": [
                                                    {
                                                        "type": "scale",
                                                        "scale": [1.0]
                                                        + (
                                                            np.array(self.voxel_size)
                                                            * mag.mag.to_np()
                                                        ).tolist(),
                                                    }
                                                ],
                                            }
                                            for mag in layer.mags.values()
                                        ],
                                    }
                                ]
                            },
                            outfile,
                            indent=4,
                        )

    def _initialize_layer_from_properties(self, properties: LayerProperties) -> Layer:
        if properties.category == COLOR_CATEGORY:
            return Layer(self, properties)
        elif properties.category == SEGMENTATION_CATEGORY:
            return SegmentationLayer(self, properties)
        else:
            raise RuntimeError(
                f"Failed to initialize layer: the specified category ({properties.category}) does not exist."
            )

    @staticmethod
    def get_remote_datasets(
        organization_id: Optional[str] = None,
        tags: Optional[Union[str, Sequence[str]]] = None,
    ) -> Mapping[str, "RemoteDataset"]:
        """
        Returns a dict of all remote datasets visible for selected organization, or the organization of the logged in user by default.
        The dict contains lazy-initialized `RemoteDataset` values for keys indicating the dataset name.

        ```python
        import webknossos as wk

        print(sorted(wk.Dataset.get_remote_datasets()))

        ds = wk.Dataset.get_remote_datasets(
            organization_id="scalable_minds"
        )["l4dense_motta_et_al_demo"]
        ```
        """
        return RemoteDatasetRegistry(organization_id=organization_id, tags=tags)


class RemoteDataset(Dataset):
    """Representation of a dataset on the webknossos server, returned from `Dataset.open_remote()`.
    Read-only image data is streamed from the webknossos server using the same interface as `Dataset`.
    Additionally, metadata can be set via the additional properties below."""

    def __init__(
        self,
        dataset_path: UPath,
        dataset_name: str,
        organization_id: str,
        sharing_token: Optional[str],
        context: ContextManager,
    ) -> None:
        """Do not call manually, please use `Dataset.open_remote()` instead."""
        try:
            super().__init__(
                dataset_path,
                voxel_size=_UNSPECIFIED_SCALE_FROM_OPEN,
                exist_ok=True,
                read_only=True,
            )
        except FileNotFoundError:
            warnings.warn(
                f"Cannot open remote webknossos dataset {dataset_path} as zarr. "
                + "Returning a stub dataset instead, accessing metadata properties might still work.",
                RuntimeWarning,
            )
            self.path = None  # type: ignore[assignment]
        self._dataset_name = dataset_name
        self._organization_id = organization_id
        self._sharing_token = sharing_token
        self._context = context

    @classmethod
    def open(cls, dataset_path: Union[str, PathLike]) -> "Dataset":
        """Do not call manually, please use `Dataset.open_remote()` instead."""
        raise RuntimeError("Please use Dataset.open_remote() instead.")

    def __repr__(self) -> str:
        return f"RemoteDataset({repr(self.url)})"

    @property
    def url(self) -> str:
        from webknossos.client.context import _get_context

        with self._context:
            wk_url = _get_context().url
        return f"{wk_url}/datasets/{self._organization_id}/{self._dataset_name}"

    def _get_dataset_info(self) -> "DatasetInfoResponse200":
        from webknossos.client._generated.api.default import dataset_info
        from webknossos.client.context import _get_generated_client

        with self._context:
            dataset_info_response = dataset_info.sync_detailed(
                organization_name=self._organization_id,
                data_set_name=self._dataset_name,
                client=_get_generated_client(),
                sharing_token=self._sharing_token,
            )
            assert dataset_info_response.status_code == 200, dataset_info_response
            parsed = dataset_info_response.parsed
            assert parsed is not None

            return parsed

    def _update_dataset_info(
        self,
        display_name: Optional[str] = _UNSET,
        description: Optional[str] = _UNSET,
        is_public: bool = _UNSET,
        tags: List[str] = _UNSET,
    ) -> None:
        from webknossos.client._generated.api.default import dataset_update
        from webknossos.client._generated.models.dataset_update_json_body import (
            DatasetUpdateJsonBody,
        )
        from webknossos.client.context import _get_generated_client

        # Atm, the wk backend needs to get previous parameters passed
        # (this is a race-condition with parallel updates).

        info = self._get_dataset_info().to_dict()
        if display_name is not _UNSET:
            info["displayName"] = display_name
        if description is not _UNSET:
            info["description"] = description
        if tags is not _UNSET:
            info["tags"] = tags
        if is_public is not _UNSET:
            info["isPublic"] = is_public
        if display_name is not _UNSET:
            info["displayName"] = display_name

        with self._context:
            dataset_info_update_response = dataset_update.sync_detailed(
                organization_name=self._organization_id,
                data_set_name=self._dataset_name,
                client=_get_generated_client(),
                json_body=DatasetUpdateJsonBody.from_dict(info),
            )
            assert (
                dataset_info_update_response.status_code == 200
            ), dataset_info_update_response

    @property
    def display_name(self) -> Optional[str]:
        return self._get_dataset_info().display_name

    @display_name.setter
    def display_name(self, display_name: Optional[str]) -> None:
        self._update_dataset_info(display_name=display_name)

    @display_name.deleter
    def display_name(self) -> None:
        self.display_name = None

    @property
    def description(self) -> Optional[str]:
        return self._get_dataset_info().description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        self._update_dataset_info(description=description)

    @description.deleter
    def description(self) -> None:
        self.description = None

    @property
    def tags(self) -> Tuple[str, ...]:
        return tuple(self._get_dataset_info().tags)

    @tags.setter
    def tags(self, tags: Sequence[str]) -> None:
        self._update_dataset_info(tags=list(tags))

    @property
    def is_public(self) -> bool:
        return bool(self._get_dataset_info().is_public)

    @is_public.setter
    def is_public(self, is_public: bool) -> None:
        self._update_dataset_info(is_public=is_public)

    @property
    def sharing_token(self) -> str:
        from webknossos.client._generated.api.default import dataset_sharing_token
        from webknossos.client.context import _get_generated_client

        with self._context:
            dataset_sharing_token_response = dataset_sharing_token.sync_detailed(
                organization_name=self._organization_id,
                data_set_name=self._dataset_name,
                client=_get_generated_client(),
            )
            assert (
                dataset_sharing_token_response.status_code == 200
            ), dataset_sharing_token_response
            assert dataset_sharing_token_response.parsed is not None
            return dataset_sharing_token_response.parsed.sharing_token

    @property
    def allowed_teams(self) -> Tuple["Team", ...]:
        from webknossos.administration.user import Team

        return tuple(
            Team(id=i.id, name=i.name, organization_id=i.organization)
            for i in self._get_dataset_info().allowed_teams
        )

    @allowed_teams.setter
    def allowed_teams(self, allowed_teams: Sequence[Union[str, "Team"]]) -> None:
        from webknossos.administration.user import Team
        from webknossos.client._generated.api.default import dataset_update_teams
        from webknossos.client.context import _get_generated_client

        team_ids = [i.id if isinstance(i, Team) else i for i in allowed_teams]

        with self._context:
            dataset_update_teams_response = dataset_update_teams.sync_detailed(
                organization_name=self._organization_id,
                data_set_name=self._dataset_name,
                client=_get_generated_client(),
                json_body=team_ids,
            )
            assert (
                dataset_update_teams_response.status_code == 200
            ), dataset_update_teams_response
