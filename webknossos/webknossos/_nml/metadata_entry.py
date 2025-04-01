from collections.abc import Sequence
from typing import Literal, NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import as_float_unless_none, as_int_unless_none, enforce_not_null


class MetadataEntry(NamedTuple):
    key: str
    type: Literal["str", "number", "list"]
    value: str | int | float | Sequence[str]

    def _dump(self, xf: XmlWriter) -> None:
        if self.type == "str":
            xf.tag(
                "metadataEntry",
                {"key": self.key, "stringValue": self.value},
            )
        elif self.type == "number":
            xf.tag(
                "metadataEntry",
                {"key": self.key, "numberValue": str(self.value)},
            )
        elif self.type == "list":
            assert isinstance(self.value, Sequence)
            xf.tag(
                "metadataEntry",
                {
                    "key": self.key,
                    **{f"stringListValue-{i}": val for i, val in enumerate(self.value)},
                },
            )

    @classmethod
    def _parse(cls, nml_metadata_entry: Element) -> "MetadataEntry":
        if nml_metadata_entry.get("stringValue") is not None:
            return cls(
                enforce_not_null(nml_metadata_entry.get("key", default=None)),
                "str",
                enforce_not_null(nml_metadata_entry.get("stringValue", default=None)),
            )
        elif nml_metadata_entry.get("numberValue") is not None:
            number_value: int | float | None
            try:
                number_value = as_int_unless_none(nml_metadata_entry.get("numberValue"))
            except ValueError:
                number_value = as_float_unless_none(
                    nml_metadata_entry.get("numberValue")
                )
            return cls(
                enforce_not_null(nml_metadata_entry.get("key", default=None)),
                "number",
                enforce_not_null(number_value),
            )
        elif nml_metadata_entry.get("stringListValue-0") is not None:
            string_list = []
            i = 0
            while nml_metadata_entry.get(f"stringListValue-{i}") is not None:
                string_list.append(
                    enforce_not_null(nml_metadata_entry.get(f"stringListValue-{i}"))
                )
                i += 1
            return cls(
                enforce_not_null(nml_metadata_entry.get("key", default=None)),
                "list",
                string_list,
            )
        else:
            raise ValueError("Invalid metadata entry")

    def __repr__(self) -> str:
        return f"{self.key} ({self.type}): '{self.value}'"
