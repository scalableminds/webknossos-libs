from os import PathLike
from pathlib import Path

import attr
from upath import UPath

from webknossos.client.api_client.models import ApiMeshAdHoc
from webknossos.client.api_client.tracingstore_api_client import TracingstoreApiClient
from webknossos.geometry.mag import Mag


@attr.frozen
class Tracingstore:
    """Tracingstore class for interactions with the tracing store."""

    name: str
    url: str
    _api_client: TracingstoreApiClient

    @classmethod
    def get_tracingstore(
        cls,
    ) -> "Tracingstore":
        """Get the tracingstore for current webknossos url.


        Returns:
            Tracingstore object

        Examples:
            ```
            # Get a list of all datastores that allow dataset uploads
            tracingstore = Tracingstore.get_tracingstore()
            ```
        """

        from ..client.context import _get_context

        context = _get_context()
        api_tracingstore = context.api_client_with_auth.tracingstore()
        api_client = context.get_tracingstore_api_client()
        return cls(api_tracingstore.name, api_tracingstore.url, api_client)

    def download_mesh(
        self,
        tracing_id: str,
        segment_id: int,
        download_path: PathLike | str,
        token: str | None = None,
    ) -> Path:
        """Download a mesh from the tracing store.
        Args:
            tracing_id: ID of the tracing
            download_path: Path to save the downloaded mesh
            token: Optional token for authentication
        Returns:
            Path to the downloaded mesh file
        Examples:
            ```
            # Download a mesh from the tracing store
            tracingstore = Tracingstore.get_tracingstore()
            mesh_path = tracingstore.download_mesh(tacing_id="tracing_id", segment_id=2, download_path="path/to/save/mesh.stl")
            ```
        """
        download_path = UPath(download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        mesh = ApiMeshAdHoc(
            lod=0,
            segment_id=segment_id,
            mapping_name=None,
            mapping_type="agglomerate",
            mag=Mag(1).to_tuple(),
            seed_position=(0, 0, 0),
        )
        with download_path.open("wb") as file:
            for chunk in self._api_client.annotation_download_mesh(
                mesh, tracing_id, token
            ):
                file.write(chunk)

        return download_path
