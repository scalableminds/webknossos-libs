from typing import NamedTuple

from webknossos.client.api_client.models import (
    ApiLinkedLayerIdentifier,
    ApiLinkedLayerIdentifierLegacy,
)
from webknossos.client.context import _get_context
from webknossos.dataset.layer import RemoteLayer


class LayerToLink(NamedTuple):
    dataset_id: str
    layer_name: str
    new_layer_name: str | None = None
    organization_id: str | None = (
        None  # defaults to the user's organization before uploading
    )

    def as_api_linked_layer_identifier(self) -> ApiLinkedLayerIdentifier:
        assert self.dataset_id is not None, f"The dataset id is not set: {self}"
        return ApiLinkedLayerIdentifier(
            self.dataset_id,
            self.layer_name,
            self.new_layer_name,
        )

    def as_api_linked_layer_identifier_legacy(self) -> ApiLinkedLayerIdentifierLegacy:
        context = _get_context()
        return ApiLinkedLayerIdentifierLegacy(
            self.organization_id or context.organization_id,
            #  webknossos checks for id too, if the name cannot be found
            self.dataset_id,
            self.layer_name,
            self.new_layer_name,
        )

    @classmethod
    def from_remote_layer(cls, remote_layer: RemoteLayer) -> "LayerToLink":
        return cls(
            remote_layer.dataset.dataset_id,
            remote_layer.name,
            new_layer_name=None,
        )
