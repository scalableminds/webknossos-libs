from os import PathLike
from typing import TYPE_CHECKING, Literal

from upath import UPath

from webknossos.client.api_client.models import ApiAdHocMeshInfo, ApiPrecomputedMeshInfo
from webknossos.dataset.layer.remote_layer import RemoteLayer
from webknossos.dataset.layer.segmentation_layer import (
    RemoteAttachments,
)
from webknossos.dataset.layer.segmentation_layer.abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)
from webknossos.dataset_properties import SegmentationLayerProperties
from webknossos.geometry import Vec3Int
from webknossos.geometry.mag import Mag, MagLike

if TYPE_CHECKING:
    from webknossos.dataset import RemoteDataset


class RemoteSegmentationLayer(
    AbstractSegmentationLayer[RemoteAttachments], RemoteLayer
):
    def __init__(
        self,
        dataset: "RemoteDataset",
        properties: SegmentationLayerProperties,
        read_only: bool,
    ) -> None:
        super().__init__(dataset, properties, read_only)

    @property
    def _AttachmentsType(self) -> type[RemoteAttachments]:
        return RemoteAttachments

    def download_mesh(
        self,
        segment_id: int,
        output_dir: PathLike | UPath | str,
        mesh_file_name: str | None = None,
        datastore_url: str | None = None,
        lod: int = 0,
        mapping_name: str | None = None,
        mapping_type: Literal["agglomerate", "json"] | None = None,
        mag: MagLike | None = None,
        seed_position: Vec3Int | None = None,
        token: str | None = None,
    ) -> UPath:
        from webknossos.client.context import _get_context
        from webknossos.datastore import Datastore

        context = _get_context()
        datastore_url = datastore_url or Datastore.get_upload_url()
        mesh_info: ApiAdHocMeshInfo | ApiPrecomputedMeshInfo
        if mesh_file_name is not None:
            mesh_info = ApiPrecomputedMeshInfo(
                lod=lod,
                mesh_file_name=mesh_file_name,
                segment_id=segment_id,
                mapping_name=mapping_name,
            )
        else:
            assert mag is not None, "mag is required for downloading ad-hoc mesh"
            assert seed_position is not None, (
                "seed_position is required for downloading ad-hoc mesh"
            )
            mesh_info = ApiAdHocMeshInfo(
                lod=lod,
                segment_id=segment_id,
                mapping_name=mapping_name,
                mapping_type=mapping_type,
                mag=Mag(mag).to_tuple(),
                seed_position=seed_position.to_tuple(),
            )
        file_path: UPath
        datastore = context.get_datastore_api_client(datastore_url=datastore_url)
        api_dataset = context.api_client.dataset_info(
            dataset_id=self.dataset.dataset_id
        )
        directory_name = api_dataset.directory_name
        mesh_download = datastore.download_mesh(
            mesh_info=mesh_info,
            dataset_id=self.dataset.dataset_id,
            layer_name=self.name,
            token=token,
        )
        file_path = UPath(output_dir) / f"{directory_name}_{self.name}_{segment_id}.stl"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            for chunk in mesh_download:
                f.write(chunk)
        return file_path
