from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    MutableMapping,
    Sequence,
    TypeVar,
    Union,
)

from webknossos.client.api_client.models import ApiDataset, ApiFolder, ApiMetadata
from webknossos.utils import infer_metadata_type, parse_metadata_value

_T = TypeVar("_T", bound="Metadata")


class Metadata(MutableMapping):
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
        self._has_changed: bool = False
        self._mapping: Dict = {}

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
                self._mapping = {
                    element.key: parse_metadata_value(element.value, element.type)
                    for element in metadata
                }
            else:
                self._mapping = {}
            yield self
        finally:
            if self._has_changed:
                api_metadata = [
                    ApiMetadata(key=k, type=infer_metadata_type(v), value=v)
                    for k, v in self._mapping.items()
                ]

                full_object.metadata = api_metadata
                if self._api_type == ApiDataset:
                    client._patch_json(f"{self._api_path}{self._id}", full_object)
                else:
                    client._put_json(f"{self._api_path}{self._id}", full_object)
                self._has_changed = False

    def __setitem__(
        self, key: str, value: Union[str, int, float, Sequence[str]]
    ) -> None:
        with self._recent_metadata() as metadata:
            metadata._has_changed = True
            metadata._mapping[key] = value

    def __getitem__(self, key: str) -> Union[str, int, float, Sequence[str]]:
        with self._recent_metadata() as metadata:
            return metadata._mapping[key]

    def __delitem__(self, key: str) -> None:
        with self._recent_metadata() as metadata:
            metadata._has_changed = True
            del metadata._mapping[key]

    def __contains__(self, key: object) -> bool:
        with self._recent_metadata() as metadata:
            return key in metadata._mapping

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Metadata):
            raise NotImplementedError(
                f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
            )
        with self._recent_metadata() as metadata:
            return metadata._mapping == other._mapping

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __iter__(self) -> Iterator[Any]:
        with self._recent_metadata() as metadata:
            return iter(metadata._mapping)

    def __len__(self) -> int:
        with self._recent_metadata() as metadata:
            return len(metadata._mapping)

    def __repr__(self) -> str:
        with self._recent_metadata() as metadata:
            return f"{self.__class__.__name__}({repr(metadata._mapping)})"


class FolderMetadata(Metadata):
    _api_path = "/folders/"
    _api_type = ApiFolder


class DatasetMetadata(Metadata):
    _api_path = "/datasets/"
    _api_type = ApiDataset
