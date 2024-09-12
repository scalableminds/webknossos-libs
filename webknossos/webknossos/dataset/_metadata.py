from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Sequence,
    TypeVar,
    Union,
)

from webknossos.client.api_client.models import ApiDataset, ApiFolder, ApiMetadata
from webknossos.utils import infer_metadata_type, parse_metadata_value

_T = TypeVar("_T", bound="Metadata")


class Metadata(dict):
    __slots__ = ()
    _api_path: str
    _api_type: Any

    def __init__(self, _id: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        if not self._api_path or not self._api_type:
            raise NotImplementedError(
                "This class is not meant to be used directly. Please use FolderMetadata or DatasetMetadata."
            )
        super().__init__(*args, **kwargs)
        self._id: str = _id

    @contextmanager
    def _recent_metadata(self: _T) -> Generator[_T, None, None]:
        from ..client.context import _get_api_client

        try:
            client = _get_api_client()
            full_object = client._get_json(
                f"{self._api_path}{self._id}",
                self._api_type,  # type: ignore
            )
            metadata: List[ApiMetadata] = full_object.metadata
            if metadata is not None:
                self = self.__class__(
                    self._id,
                    {
                        element.key: parse_metadata_value(element.value, element.type)
                        for element in metadata
                    },
                )
            else:
                self = self.__class__(self._id)
            yield self
        finally:
            api_metadata = [
                ApiMetadata(key=k, type=infer_metadata_type(v), value=v)
                for k, v in self.items()
            ]

            full_object.metadata = api_metadata
            if self._api_type == ApiDataset:
                client._patch_json(f"{self._api_path}{self._id}", full_object)
            else:
                client._put_json(f"{self._api_path}{self._id}", full_object)

    def __setitem__(
        self, key: str, value: Union[str, int, float, Sequence[str]]
    ) -> None:
        with self._recent_metadata() as metadata:
            super(Metadata, metadata).__setitem__(key, value)

    def __getitem__(self, key: str) -> Union[str, int, float, Sequence[str]]:
        with self._recent_metadata() as metadata:
            return super(Metadata, metadata).__getitem__(key)

    def __delitem__(self, key: str) -> None:
        with self._recent_metadata() as metadata:
            super(Metadata, metadata).__delitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._recent_metadata() as metadata:
            return super(Metadata, metadata).__contains__(key)

    def __iter__(self) -> Iterator[Any]:
        with self._recent_metadata() as metadata:
            return super(Metadata, metadata).__iter__()

    def __len__(self) -> int:
        with self._recent_metadata() as metadata:
            return super(Metadata, metadata).__len__()

    def __repr__(self) -> str:
        with self._recent_metadata() as metadata:
            return f"{self.__class__.__name__}({super(Metadata, metadata).__repr__()})"


class FolderMetadata(Metadata):
    _api_path = "/folders/"
    _api_type = ApiFolder


class DatasetMetadata(Metadata):
    _api_path = "/datasets/"
    _api_type = ApiDataset
