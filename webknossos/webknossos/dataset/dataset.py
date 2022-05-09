import copy
import inspect
import json
import logging
import re
import warnings
from argparse import Namespace
from contextlib import nullcontext
from os import PathLike
from os.path import relpath
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import attr
import numpy as np
from boltons.typeutils import make_sentinel
from upath import UPath

from ..geometry.vec3_int import Vec3IntLike
from ._array import ArrayException, ArrayInfo, BaseArray, DataFormat

if TYPE_CHECKING:
    from ..client._generated.models import DatasetInfoResponse200
    from ..client._upload_dataset import LayerToLink
    from ..administration.user import Team

from ..geometry import BoundingBox, Mag
from ..utils import (
    copy_directory_with_symlinks,
    copytree,
    get_executor_for_args,
    is_fs_path,
    rmtree,
    warn_deprecated,
)
from ._utils.infer_bounding_box_existing_files import infer_bounding_box_existing_files
from .layer import (
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
from .view import View

logger = logging.getLogger(__name__)

DEFAULT_BIT_DEPTH = 8
DEFAULT_DATA_FORMAT = DataFormat.WKW
PROPERTIES_FILE_NAME = "datasource-properties.json"

_DATASET_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/datasets/"
    + r"(?P<organization_id>[^/]*)/(?P<dataset_name>[^/]*)(/(view)?)?"
    + r"(\?token=(?P<sharing_token>[^#]*))?"
)


def _copy_job(args: Tuple[View, View, int]) -> None:
    (source_view, target_view, _) = args
    # Copy the data form one view to the other in a buffered fashion
    target_view.write(source_view.read())


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
        Currently exist_ok=True is the deprecated default and will change in future releases.
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

    def upload(
        self,
        new_dataset_name: Optional[str] = None,
        layers_to_link: Optional[List["LayerToLink"]] = None,
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

        from webknossos.client._upload_dataset import upload_dataset

        return self.open_remote(
            upload_dataset(self, new_dataset_name, layers_to_link, jobs)
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
        dtype_per_layer: Optional[Union[str, np.dtype, type]] = None,
        dtype_per_channel: Optional[Union[str, np.dtype, type]] = None,
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
                    largest_segment_id=kwargs["largest_segment_id"],
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
        dtype_per_layer: Optional[Union[str, np.dtype, type]] = None,
        dtype_per_channel: Optional[Union[str, np.dtype, type]] = None,
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
        if layer_path.is_symlink():
            layer_path.unlink()
        else:
            # rmtree does not recurse into linked dirs, but removes the link
            rmtree(layer_path)
        self._export_as_json()

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

        if isinstance(foreign_layer, Layer):
            foreign_layer_path = foreign_layer.path
        else:
            foreign_layer_path = UPath(foreign_layer)

        foreign_layer_name = foreign_layer_path.name
        layer_name = (
            new_layer_name if new_layer_name is not None else foreign_layer_name
        )
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

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

        (self.path / layer_name).symlink_to(foreign_layer_symlink_path)
        original_layer = Dataset.open(foreign_layer_path.parent).get_layer(
            foreign_layer_name
        )
        layer_properties = copy.deepcopy(original_layer._properties)
        layer_properties.name = layer_name
        self._properties.data_layers += [layer_properties]
        self._layers[layer_name] = self._initialize_layer_from_properties(
            layer_properties
        )

        self._export_as_json()
        return self.layers[layer_name]

    def add_copy_layer(
        self,
        foreign_layer: Union[str, Path, Layer],
        new_layer_name: Optional[str] = None,
    ) -> Layer:
        """
        Copies the data at `foreign_layer` which belongs to another dataset to the current dataset.
        Additionally, the relevant information from the `datasource-properties.json` of the other dataset are copied too.
        If new_layer_name is None, the name of the foreign layer is used.
        """
        self._ensure_writable()

        if isinstance(foreign_layer, Layer):
            foreign_layer_path = foreign_layer.path
        else:
            foreign_layer_path = UPath(foreign_layer)

        foreign_layer_name = foreign_layer_path.name
        layer_name = (
            new_layer_name if new_layer_name is not None else foreign_layer_name
        )
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot copy {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

        copytree(foreign_layer_path, self.path / layer_name)
        original_layer = Dataset.open(foreign_layer_path.parent).get_layer(
            foreign_layer_name
        )
        layer_properties = copy.deepcopy(original_layer._properties)
        layer_properties.name = layer_name
        self._properties.data_layers += [layer_properties]
        self._layers[layer_name] = self._initialize_layer_from_properties(
            layer_properties
        )

        self._export_as_json()
        return self.layers[layer_name]

    def copy_dataset(
        self,
        new_dataset_path: Union[str, Path],
        voxel_size: Optional[Tuple[float, float, float]] = None,
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        data_format: Optional[Union[str, DataFormat]] = None,
        compress: Optional[bool] = None,
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
        args: Optional[Namespace] = None,
    ) -> "Dataset":
        """
        Creates a new dataset at `new_dataset_path` and copies the data from the current dataset to `empty_target_ds`.
        If not specified otherwise, the `voxel_size`, `chunk_size`, `chunks_per_shard` and `compress` of the current dataset
        are also used for the new dataset. The method also accepts the parameters `block_len` and `file_size`,
        which were deprecated by `chunk_size` and `chunks_per_shard`.
        WKW layers can only be copied to datasets on local file systems.
        """

        chunk_size, chunks_per_shard = _get_sharding_parameters(
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
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

        with get_executor_for_args(args) as executor:
            for layer_name, layer in self.layers.items():
                new_ds_properties = copy.deepcopy(
                    self.get_layer(layer_name)._properties
                )
                # Initializing a layer with non-empty mags requires that the files on disk already exist.
                # The MagViews are added manually afterwards
                new_ds_properties.mags = []
                if data_format is not None:
                    new_ds_properties.data_format = DataFormat(data_format)
                new_ds._properties.data_layers += [new_ds_properties]
                target_layer = new_ds._initialize_layer_from_properties(
                    new_ds_properties
                )
                new_ds._layers[layer_name] = target_layer

                bbox = self.get_layer(layer_name).bounding_box

                for mag, mag_view in layer.mags.items():
                    chunk_size = chunk_size or mag_view.info.chunk_size
                    compression_mode = compress or mag_view.info.compression_mode
                    chunks_per_shard = (
                        chunks_per_shard or mag_view.info.chunks_per_shard
                    )
                    target_mag = target_layer.add_mag(
                        mag, chunk_size, chunks_per_shard, compression_mode
                    )

                    target_layer.bounding_box = bbox

                    # The data gets written to the target_mag.
                    # Therefore, the chunk size is determined by the target_mag to prevent concurrent writes
                    mag_view.for_zipped_chunks(
                        func_per_chunk=_copy_job,
                        target_view=target_mag.get_view(),
                        executor=executor,
                        progress_desc=f"Copying mag {mag.to_layer_name()} from layer {layer_name}",
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
                ignore=[str(mag) for mag in layer.mags] + [PROPERTIES_FILE_NAME],
                make_relative=make_relative,
            )

        return new_dataset

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

    def _initialize_layer_from_properties(self, properties: LayerProperties) -> Layer:
        if properties.category == COLOR_CATEGORY:
            return Layer(self, properties)
        elif properties.category == SEGMENTATION_CATEGORY:
            return SegmentationLayer(self, properties)
        else:
            raise RuntimeError(
                f"Failed to initialize layer: the specified category ({properties.category}) does not exist."
            )


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
        """Do not call manually, please use `Dataset.remote_open()` instead."""
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
        """Do not call manually, please use `Dataset.remote_open()` instead."""
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
