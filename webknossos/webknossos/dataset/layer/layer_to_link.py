from typing import TYPE_CHECKING, NamedTuple

from webknossos.client.api_client.models import (
    ApiLinkedLayerIdentifier,
)

if TYPE_CHECKING:
    from webknossos.dataset.layer import RemoteLayer


class LayerToLink(NamedTuple):
    dataset_id: str
    layer_name: str
    new_layer_name: str | None = None
    organization_id: str | None = (
        None  # defaults to the user's organization before uploading
    )

    def _as_api_linked_layer_identifier(self) -> ApiLinkedLayerIdentifier:
        assert self.dataset_id is not None, f"The dataset id is not set: {self}"
        return ApiLinkedLayerIdentifier(
            self.dataset_id,
            self.layer_name,
            self.new_layer_name,
        )

    @classmethod
    def from_remote_layer(
        cls, remote_layer: "RemoteLayer", new_layer_name: str | None = None
    ) -> "LayerToLink":
        return cls(
            remote_layer.dataset.dataset_id,
            remote_layer.name,
            new_layer_name=new_layer_name,
        )
