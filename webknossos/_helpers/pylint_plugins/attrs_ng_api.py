# Re-Licensed under GPLv3, see https://www.gnu.org/licenses/gpl-3.0.html

# Taken from the original PR https://github.com/PyCQA/astroid/pull/839

import astroid


def register(linter):
    # Needed for registering the plugin.
    del linter


ATTRIB_NAMES = frozenset(("attr.ib", "attrib", "attr.attrib", "attr.field", "field"))
ATTRS_NAMES = frozenset(
    (
        "attr.s",
        "attrs",
        "attr.attrs",
        "attr.attributes",
        "attr.define",
        "attr.mutable",
        "attr.frozen",
    )
)


def is_decorated_with_attrs(node, decorator_names=ATTRS_NAMES):
    """Return True if a decorated node has
    an attr decorator applied."""
    if not node.decorators:
        return False
    for decorator_attribute in node.decorators.nodes:
        if isinstance(decorator_attribute, astroid.Call):  # decorator with arguments
            decorator_attribute = decorator_attribute.func
        if decorator_attribute.as_string() in decorator_names:
            return True
    return False


def attr_attributes_transform(node):
    """Given that the ClassNode has an attr decorator,
    rewrite class attributes as instance attributes
    """
    # Astroid can't infer this attribute properly
    # Prevents https://github.com/PyCQA/pylint/issues/1884
    node.locals["__attrs_attrs__"] = [astroid.Unknown(parent=node)]

    for cdefbodynode in node.body:
        if not isinstance(cdefbodynode, (astroid.Assign, astroid.AnnAssign)):
            continue
        if isinstance(cdefbodynode.value, astroid.Call):
            if cdefbodynode.value.func.as_string() not in ATTRIB_NAMES:
                continue
        else:
            continue
        targets = (
            cdefbodynode.targets
            if hasattr(cdefbodynode, "targets")
            else [cdefbodynode.target]
        )
        for target in targets:

            rhs_node = astroid.Unknown(
                lineno=cdefbodynode.lineno,
                col_offset=cdefbodynode.col_offset,
                parent=cdefbodynode,
            )
            node.locals[target.name] = [rhs_node]
            node.instance_attrs[target.name] = [rhs_node]


astroid.MANAGER.register_transform(
    astroid.ClassDef, attr_attributes_transform, is_decorated_with_attrs
)
