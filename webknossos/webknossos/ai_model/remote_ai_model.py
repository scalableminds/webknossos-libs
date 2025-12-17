from upath import UPath

from webknossos.client import _get_context
from webknossos.client.api_client.models import ApiReserveAiModelUploadToPathParameters

from ..dataset.transfer_mode import TransferMode
from .ai_model_category import AiModelCategory


class RemoteAiModel:
    def __init__(self, ai_model_id: str, name: str, path: str, is_usable: bool) -> None:
        """A remote AiModel instance.
        Note: Please not initialize this class directly, use RemoteAiModel.open_remote() instead."""
        self.ai_model_id = ai_model_id
        self.name = name
        self.path = path
        self.is_usable = is_usable

    @classmethod
    def open_remote(cls, ai_model_id: str) -> "RemoteAiModel":
        context = _get_context()
        api_ai_model = context.api_client.get_ai_model_info(ai_model_id)
        return RemoteAiModel(
            ai_model_id, api_ai_model.name, api_ai_model.path, api_ai_model.is_usable
        )

    @classmethod
    def upload_from_path(
        cls,
        src_path: UPath,
        existing_id: str | None,
        data_store_name: str,
        name: str,
        comment: str | None,
        category: AiModelCategory | None,
        path_prefix: str | None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> "RemoteAiModel":
        if transfer_mode == TransferMode.HTTP:
            raise ValueError("HTTP transfer mode is not supported for this method")

        api_client = _get_context().api_client

        target_info = api_client.reserve_ai_model_upload_to_path(
            ApiReserveAiModelUploadToPathParameters(
                existing_id,
                data_store_name,
                name,
                comment,
                category,
                path_prefix,
            )
        )

        transfer_mode.transfer(src_path, UPath(target_info.path))

        api_client.finish_ai_model_upload_to_path(target_info.id)

        return cls.open_remote(target_info.id)
