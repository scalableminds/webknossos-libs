from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import attr
from boltons.strutils import unit_len

import webknossos._nml as wknml

if TYPE_CHECKING:
    from .volume_layer import SegmentInformation, VolumeLayer


@attr.define()
class SegmentGroup:
    """A hierarchical container for organizing segments of a volume layer.

    Segment groups can be nested. A segment is assigned to a group by setting
    `SegmentInformation.group`; segments with `group=None` are top-level.

    Attributes:
        name: Name of the group.
        id: Read-only unique identifier for the group within its volume layer.
        groups: Iterator over immediate subgroups.
        segments: Immediate segments of this group.

    Examples:
        ```python
        volume_layer = annotation.get_volume_layer("segmentation")
        neuron = volume_layer.add_segment_group("neuron_42")
        dendrites = neuron.add_group("dendrites")

        volume_layer.segments[1] = wk.SegmentInformation(
            name="dendrite fragment",
            anchor_position=None,
            color=None,
            metadata={},
            group=dendrites,
        )

        print(neuron.flattened_segments())
        ```

    Notes:
        - Groups should be created with `VolumeLayer.add_segment_group` or `add_group`
          rather than directly.

    See Also:
        - VolumeLayer: Volume annotation layer owning the groups and segments
        - SegmentInformation: Per-segment information, including its group
    """

    _id: int = attr.ib(init=False)
    name: str
    _child_groups: list["SegmentGroup"] = attr.ib(
        factory=list,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'child group')}>",
    )
    _volume_layer: "VolumeLayer" = attr.ib(eq=False, repr=False)
    _enforced_id: int | None = attr.ib(None, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        id_generator = self._volume_layer._segment_group_id_generator
        if self._enforced_id is not None:
            self._id = self._enforced_id
            id_generator.register(self._id)
        else:
            self._id = id_generator.__next__()

    @property
    def id(self) -> int:
        """Read-only property."""
        return self._id

    @property
    def groups(self) -> Iterator["SegmentGroup"]:
        """Returns all (immediate) group children as an iterator.
        Use flattened_groups if you also need groups within subgroups."""
        return iter(self._child_groups)

    def add_group(
        self,
        name: str,
        _enforced_id: int | None = None,
    ) -> "SegmentGroup":
        """Creates and adds a new subgroup to this group.

        Args:
            name: Name for the new group.
            _enforced_id: Optional specific ID for the group (internal use).

        Returns:
            SegmentGroup: The newly created group.
        """
        new_group = SegmentGroup(
            name, volume_layer=self._volume_layer, enforced_id=_enforced_id
        )
        self._child_groups.append(new_group)
        return new_group

    def flattened_groups(self) -> Iterator["SegmentGroup"]:
        """Returns an iterator of all groups within this group (and its subgroups)."""
        for group in self.groups:
            yield group
            yield from group.flattened_groups()

    def get_group_by_id(self, group_id: int) -> "SegmentGroup":
        """Returns the group within this group (and its subgroups) which has the specified id."""
        for group in self.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No segment group with id {group_id} was found")

    @property
    def segments(self) -> dict[int, "SegmentInformation"]:
        """Returns the (immediate) segments of this group, keyed by segment id.
        Use flattened_segments if you also need segments within subgroups."""
        group = None if self._is_root() else self
        return {
            segment_id: segment_info
            for segment_id, segment_info in self._volume_layer.segments.items()
            if segment_info.group == group
        }

    def flattened_segments(self) -> dict[int, "SegmentInformation"]:
        """Returns all segments of this group and its subgroups, keyed by segment id."""
        if self._is_root():
            return dict(self._volume_layer.segments)
        group_ids = {self.id} | {group.id for group in self.flattened_groups()}
        return {
            segment_id: segment_info
            for segment_id, segment_info in self._volume_layer.segments.items()
            if segment_info.group is not None and segment_info.group.id in group_ids
        }

    def as_nml_group(self) -> wknml.Group:
        """Converts this group to its NML representation."""
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._child_groups],
        )

    def _is_root(self) -> bool:
        return self is self._volume_layer._segment_group_root

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and self._id == other._id

    def __hash__(self) -> int:
        return id(self)
