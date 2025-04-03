from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import cattrs
from attrs import fields as attr_fields
from attrs import has as is_attr_class

from ...utils import snake_to_camel_case

T = TypeVar("T")

custom_converter = cattrs.Converter()

# Structuring and destructuring used for the attrs classes in apiclient.models.
# The server expects and sends camelCase fields, we want snake_case here
# However, the case conversion should happen only for the attrs classes,
# and not for dicts that may contain user data (e.g. user experiences)


def attr_to_camel_case_structure(cl: type[T]) -> Callable[[Mapping[str, Any], Any], T]:
    return cattrs.gen.make_dict_structure_fn(
        cl,
        custom_converter,
        **{
            a.name: cattrs.gen.override(rename=snake_to_camel_case(a.name))
            # https://github.com/python/mypy/issues/16254
            for a in attr_fields(cl)  # type: ignore[arg-type,misc]
        },
    )


def attr_to_camel_case_unstructure(cl: type[T]) -> Callable[[T], dict[str, Any]]:
    return cattrs.gen.make_dict_unstructure_fn(
        cl,
        custom_converter,
        **{
            a.name: cattrs.gen.override(rename=snake_to_camel_case(a.name))
            # https://github.com/python/mypy/issues/16254
            for a in attr_fields(cl)  # type: ignore[arg-type,misc]
        },
    )


custom_converter.register_structure_hook_factory(
    is_attr_class, attr_to_camel_case_structure
)
custom_converter.register_unstructure_hook_factory(
    is_attr_class, attr_to_camel_case_unstructure
)
