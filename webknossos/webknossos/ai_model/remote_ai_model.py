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
        """
        Uploads an AI model from a local or remote path to the webknossos data store.

        :param src_path: The path of the model artifact.
        """
        if transfer_mode == TransferMode.HTTP:
            raise ValueError("HTTP transfer mode is not supported for this method")

        api_client = _get_context().api_client

        target_info = api_client.reserve_ai_model_upload_to_path(
            ApiReserveAiModelUploadToPathParameters(
                existing_id,
                data_store_name,
                name,
                comment,
                category.value if category is not None else None,
                path_prefix,
            )
        )

        # this is the **parent** directory where we need to upload the model
        # after uploading, we want the following structure:
        # target_info.path/
        #   config_train_model.yaml
        #   model/
        target_path = UPath(target_info.path)
        # the parent dir needs to exist before copying the config
        target_path.mkdir(parents=True, exist_ok=True)

        # depending on whether the model was trained locally or on s3, the folder name may differ
        # also, the parent may contain unwanted folders like "logs", "cfut", etc.
        # therefor we directly transfer only the config and the model folder to the desired target structure

        # copy config_train_model.yaml
        transfer_mode.transfer(
            src_path.parent / "config_train_model.yaml",
            target_path / "config_train_model.yaml",
            progress_desc_label="AI model config",
        )

        # for uploaded models, the target folder name should always just be "model"
        transfer_mode.transfer(
            src_path,
            target_path / "model",
            progress_desc_label="AI model",
        )

        api_client.finish_ai_model_upload_to_path(target_info.id)

        return cls.open_remote(target_info.id)
