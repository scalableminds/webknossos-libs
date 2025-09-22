from typing import NamedTuple

from webknossos import Layer, RemoteDataset
from webknossos.client.api_client.models import (
    ApiLinkedLayerIdentifier,
    ApiLinkedLayerIdentifierLegacy,
)
from webknossos.client.context import _get_context


class LayerToLink(NamedTuple):
    """
    Describes an existing layer that should be linked to a new dataset.
    """

    dataset_id: str
    layer_name: str
    new_layer_name: str | None = None
    organization_id: str | None = (
        None  # defaults to the user's organization before uploading
    )

    @classmethod
    def from_remote_layer(
        cls,
        layer: Layer,
        new_layer_name: str | None = None,
        organization_id: str | None = None,
    ) -> "LayerToLink":
        ds = layer.dataset
        assert isinstance(ds, RemoteDataset), (
            f"The passed layer must belong to a RemoteDataset, but belongs to {ds}"
        )
        return cls(ds.dataset_id, layer.name, new_layer_name, organization_id)

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
