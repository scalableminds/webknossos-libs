from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.js_object_underlying import JsObjectUnderlying
    from ..models.js_object_value import JsObjectValue
    from ..models.tuple_2_string_js_value import Tuple2StringJsValue


T = TypeVar("T", bound="JsObject")


@attr.s(auto_attribs=True)
class JsObject:
    """
    Attributes:
        underlying (JsObjectUnderlying):
        fields (List['Tuple2StringJsValue']):
        value (JsObjectValue):
    """

    underlying: "JsObjectUnderlying"
    fields: List["Tuple2StringJsValue"]
    value: "JsObjectValue"
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        underlying = self.underlying.to_dict()

        fields = []
        for fields_item_data in self.fields:
            fields_item = fields_item_data.to_dict()

            fields.append(fields_item)

        value = self.value.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "underlying": underlying,
                "fields": fields,
                "value": value,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.js_object_underlying import JsObjectUnderlying
        from ..models.js_object_value import JsObjectValue
        from ..models.tuple_2_string_js_value import Tuple2StringJsValue

        d = src_dict.copy()
        underlying = JsObjectUnderlying.from_dict(d.pop("underlying"))

        fields = []
        _fields = d.pop("fields")
        for fields_item_data in _fields:
            fields_item = Tuple2StringJsValue.from_dict(fields_item_data)

            fields.append(fields_item)

        value = JsObjectValue.from_dict(d.pop("value"))

        js_object = cls(
            underlying=underlying,
            fields=fields,
            value=value,
        )

        js_object.additional_properties = d
        return js_object

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
