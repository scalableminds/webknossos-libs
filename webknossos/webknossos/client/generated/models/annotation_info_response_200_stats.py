from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="AnnotationInfoResponse200Stats")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200Stats:
    """ """

    edge_count: int
    node_count: int
    tree_count: int
    branch_point_count: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        edge_count = self.edge_count
        node_count = self.node_count
        tree_count = self.tree_count
        branch_point_count = self.branch_point_count

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "edgeCount": edge_count,
                "nodeCount": node_count,
                "treeCount": tree_count,
                "branchPointCount": branch_point_count,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        edge_count = d.pop("edgeCount")

        node_count = d.pop("nodeCount")

        tree_count = d.pop("treeCount")

        branch_point_count = d.pop("branchPointCount")

        annotation_info_response_200_stats = cls(
            edge_count=edge_count,
            node_count=node_count,
            tree_count=tree_count,
            branch_point_count=branch_point_count,
        )

        annotation_info_response_200_stats.additional_properties = d
        return annotation_info_response_200_stats

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
