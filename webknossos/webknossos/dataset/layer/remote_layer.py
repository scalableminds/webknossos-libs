from os import PathLike
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from cluster_tools import Executor
from upath import UPath

from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.dataset_properties import LayerProperties, MagViewProperties

from ...client.api_client.models import ApiReserveMagUploadToPathParameters
from ...geometry import Vec3Int
from ...geometry.mag import Mag, MagLike
from ...utils import enrich_path
from ..transfer_mode import TransferMode
from .abstract_layer import AbstractLayer, _validate_layer_name
from .view import MagView, Zarr3Config

if TYPE_CHECKING:
    from webknossos.dataset import RemoteDataset

    from .segmentation_layer import RemoteSegmentationLayer


class RemoteLayer(AbstractLayer):
    _dataset: "RemoteDataset"
    _mags: dict[Mag, MagView["RemoteLayer"]]

    def __init__(
        self, dataset: "RemoteDataset", properties: LayerProperties, read_only: bool
    ):
        super().__init__(dataset, properties, read_only)

    def _determine_read_only_and_path_for_mag(
        self, mag_properties: MagViewProperties
    ) -> tuple[bool, UPath]:
        assert mag_properties.path is not None, (
            f"Remote mags must have a path: {mag_properties}"
        )
        # In case of zarr-streaming remote datasets, the mag paths are relative to the dataset path.
        # In the case of non-streaming remote datasets, the mag paths are absolute.
        mag_path = enrich_path(mag_properties.path, self._dataset.zarr_streaming_path)
        read_only = True
        return read_only, mag_path

    @property
    def dataset(self) -> "RemoteDataset":
        return self._dataset

    def as_segmentation_layer(self) -> "RemoteSegmentationLayer":
        """Casts into SegmentationLayer."""
        from .segmentation_layer import RemoteSegmentationLayer

        if isinstance(self, RemoteSegmentationLayer):
            return self
        else:
            raise TypeError(f"self is not a SegmentationLayer. Got: {type(self)}")

    def get_mag(self, mag: MagLike) -> MagView["RemoteLayer"]:
        return super().get_mag(mag)

    def get_finest_mag(self) -> MagView["RemoteLayer"]:
        return super().get_finest_mag()

    def add_mag_as_copy(
        self,
        foreign_mag_view_or_path: PathLike | UPath | str | MagView,
        transfer_mode: TransferMode = TransferMode.COPY,
        common_storage_path_prefix: str | None = None,
        overwrite_pending: bool = True,
    ) -> MagView["RemoteLayer"]:
        """
        Copies the data at `foreign_mag_view_or_path` which can belong to another dataset
        to the current remote dataset. Additionally, the relevant information from the
        `datasource-properties.json` of the other dataset are copied, too.
        """
        self._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)

        from ...client.context import _get_api_client

        with self._dataset._context:
            client = _get_api_client()
            foreign_layer = foreign_mag_view.layer
            foreign_layer_bbox = foreign_layer.bounding_box.normalize_axes()
            reserve_parameters = ApiReserveMagUploadToPathParameters(
                layer_name=self.name,
                mag=foreign_mag_view.mag.to_list(),
                axis_order={
                    axis: index
                    for axis, index in zip(
                        foreign_layer_bbox.axes, foreign_layer_bbox.index
                    )
                },
                channel_index=foreign_layer_bbox.topleft.c
                if "c" in foreign_layer_bbox.axes
                else None,
                path_prefix=common_storage_path_prefix,
                overwrite_pending=overwrite_pending,
            )
            path = enrich_path(
                client.reserve_mag_upload_to_paths(
                    self._dataset.dataset_id, reserve_parameters
                )
            )
            transfer_mode.transfer(
                foreign_mag_view.path,
                path,
                progress_desc_label=f"{self.name}/{foreign_mag_view.mag}",
            )
            client.finish_mag_upload_to_paths(
                self._dataset.dataset_id, reserve_parameters
            )
        self._apply_server_layer_properties()
        return self.get_mag(foreign_mag_view.mag)

    def downsample(
        self,
        *,
        from_mag: Mag | None = None,
        coarsest_mag: Mag | None = None,
        interpolation_mode: str = "default",
        compress: bool | Zarr3Config = True,
        sampling_mode: str | SamplingModes = SamplingModes.ANISOTROPIC,
        align_with_other_layers: bool = True,
        buffer_shape: Vec3Int | None = None,
        force_sampling_scheme: bool = False,
        transfer_mode: TransferMode = TransferMode.COPY,
        common_storage_path_prefix: str | None = None,
        overwrite_pending: bool = True,
        executor: Executor | None = None,
    ) -> None:
        """Downsample data from a source magnification to coarser magnifications.

        Downsamples the data starting from from_mag until a magnification is >= max(coarsest_mag).
        Different sampling modes control how dimensions are downsampled.

        Note that the data is written temporarily on the local disk and uploaded afterwards so some local disk space is required.

        Args:
            from_mag (Mag | None): Source magnification to downsample from. Defaults to highest existing mag.
            coarsest_mag (Mag | None): Target magnification to stop at. Defaults to calculated value.
            interpolation_mode (str): Interpolation method to use. Defaults to "default".
                Supported modes: "median", "mode", "nearest", "bilinear", "bicubic"
            compress (bool | Zarr3Config): Whether to compress the generated magnifications. For Zarr3 datasets, codec configuration and chunk key encoding may also be supplied. Defaults to True.
            sampling_mode (str | SamplingModes): How dimensions should be downsampled.
                Defaults to ANISOTROPIC.
            align_with_other_layers (bool): Whether to align the mag selection with the dataset’s other layers. True by default.
            buffer_shape (Vec3Int | None): Shape of processing buffer. Defaults to None.
            force_sampling_scheme (bool): Force invalid sampling schemes. Defaults to False.
            transfer_mode (TransferMode). How new mags are transferred to the remote or local storage. Defaults to COPY
            common_storage_path_prefix (str | None): Optional path prefix used when transfer_mode is either COPY or MOVE_AND_SYMLINK
                                        to select one of the available WEBKNOSSOS storages.
            overwrite_pending (bool). If there are already pending/unfinished committed mags on the server, overwrite them. Defaults to True
            executor (Executor | None): Executor for parallel processing. None by default.

        Raises:
            AssertionError: If from_mag does not exist
            RuntimeError: If sampling scheme produces invalid magnifications
            AttributeError: If sampling_mode is invalid
        """

        from ..dataset import Dataset

        if from_mag is None:
            assert len(self.mags.keys()) > 0, (
                "Failed to downsample data because no existing mag was found."
            )
            from_mag = max(self.mags.keys())

        assert from_mag in self.mags.keys(), (
            f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist. Existing mags: {self.mags.keys()}."
        )
        from_mag_view = self.get_mag(from_mag)

        align_with_other_layers_dataset: bool | RemoteDataset = False
        if align_with_other_layers:
            align_with_other_layers_dataset = self.dataset

        with TemporaryDirectory() as tmpdir:
            tmp_dataset = Dataset(
                dataset_path=tmpdir,
                voxel_size_with_unit=self.dataset.voxel_size_with_unit,
            )
            tmp_layer = tmp_dataset.add_layer_like(self, self.name)
            tmp_layer.downsample(
                from_mag=from_mag,
                from_mag_view=from_mag_view,
                coarsest_mag=coarsest_mag,
                interpolation_mode=interpolation_mode,
                compress=compress,
                sampling_mode=sampling_mode,
                align_with_other_layers=align_with_other_layers_dataset,
                buffer_shape=buffer_shape,
                force_sampling_scheme=force_sampling_scheme,
                executor=executor,
            )

            for mag in tmp_layer.mags.keys():
                self.add_mag_as_copy(
                    tmp_layer.mags[mag],
                    transfer_mode=transfer_mode,
                    common_storage_path_prefix=common_storage_path_prefix,
                    overwrite_pending=overwrite_pending,
                )

    def delete_mag(self, mag: MagLike) -> None:
        self._ensure_writable()
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                f"Deleting mag {mag} failed. There is no mag with this name"
            )
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) != mag
        ]
        self._save_layer_properties()

    @property
    def name(self) -> str:
        """
        Returns the name of the layer.
        """
        return self._name

    @name.setter
    def name(self, layer_name: str) -> None:
        """
        Renames the layer to `layer_name`. This changes the layer name on the server.
        CAUTION: existing annotations that use this layer as fallback segmentation layer will break.
        """

        if layer_name == self.name:
            return
        self._ensure_metadata_writable()
        if layer_name in self.dataset.layers.keys():
            raise ValueError(
                f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
            )

        _validate_layer_name(layer_name)

        old_name = self.name
        del self.dataset._layers[self.name]
        self.dataset._layers[layer_name] = self
        self._properties.name = layer_name
        self._name: str = layer_name
        self._save_layer_properties(layer_renaming=(old_name, layer_name))

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError(
                f"Remote layer '{self.name}' is read-only, consider disabling zarr_streaming with RemoteDataset.open(use_zarr_streaming=False)"
            )

    def _apply_server_layer_properties(self) -> None:
        self.dataset._apply_server_dataset_properties()
        layer_properties = next(
            layer_properties
            for layer_properties in self._dataset._properties.data_layers
            if layer_properties.name == self.name
        )
        self._apply_properties(layer_properties, self.read_only)

    def __repr__(self) -> str:
        return f"RemoteLayer({repr(self.name)}, dtype_per_channel={self.dtype_per_channel}, num_channels={self.num_channels})"
