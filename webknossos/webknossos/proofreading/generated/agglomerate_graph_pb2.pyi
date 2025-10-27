from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class AdditionalAxisProto(_message.Message):
    __slots__ = ["bounds", "index", "name"]
    BOUNDS_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    bounds: Vec2IntProto
    index: int
    name: str
    def __init__(
        self,
        name: str | None = ...,
        index: int | None = ...,
        bounds: Vec2IntProto | _Mapping | None = ...,
    ) -> None: ...

class AdditionalCoordinateProto(_message.Message):
    __slots__ = ["name", "value"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: int
    def __init__(self, name: str | None = ..., value: int | None = ...) -> None: ...

class AgglomerateEdge(_message.Message):
    __slots__ = ["source", "target"]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    source: int
    target: int
    def __init__(self, source: int | None = ..., target: int | None = ...) -> None: ...

class AgglomerateGraph(_message.Message):
    __slots__ = ["affinities", "edges", "positions", "segments"]
    AFFINITIES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    affinities: _containers.RepeatedScalarFieldContainer[float]
    edges: _containers.RepeatedCompositeFieldContainer[AgglomerateEdge]
    positions: _containers.RepeatedCompositeFieldContainer[Vec3IntProto]
    segments: _containers.RepeatedScalarFieldContainer[int]
    def __init__(
        self,
        segments: _Iterable[int] | None = ...,
        edges: _Iterable[AgglomerateEdge | _Mapping] | None = ...,
        positions: _Iterable[Vec3IntProto | _Mapping] | None = ...,
        affinities: _Iterable[float] | None = ...,
    ) -> None: ...

class BoundingBoxProto(_message.Message):
    __slots__ = ["depth", "height", "topLeft", "width"]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TOPLEFT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    depth: int
    height: int
    topLeft: Vec3IntProto
    width: int
    def __init__(
        self,
        topLeft: Vec3IntProto | _Mapping | None = ...,
        width: int | None = ...,
        height: int | None = ...,
        depth: int | None = ...,
    ) -> None: ...

class ColorProto(_message.Message):
    __slots__ = ["a", "b", "g", "r"]
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    a: float
    b: float
    g: float
    r: float
    def __init__(
        self,
        r: float | None = ...,
        g: float | None = ...,
        b: float | None = ...,
        a: float | None = ...,
    ) -> None: ...

class ListOfVec3IntProto(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[Vec3IntProto]
    def __init__(
        self, values: _Iterable[Vec3IntProto | _Mapping] | None = ...
    ) -> None: ...

class NamedBoundingBoxProto(_message.Message):
    __slots__ = ["boundingBox", "color", "id", "isVisible", "name"]
    BOUNDINGBOX_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ISVISIBLE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    boundingBox: BoundingBoxProto
    color: ColorProto
    id: int
    isVisible: bool
    name: str
    def __init__(
        self,
        id: int | None = ...,
        name: str | None = ...,
        isVisible: bool = ...,
        color: ColorProto | _Mapping | None = ...,
        boundingBox: BoundingBoxProto | _Mapping | None = ...,
    ) -> None: ...

class Vec2IntProto(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: int | None = ..., y: int | None = ...) -> None: ...

class Vec3DoubleProto(_message.Message):
    __slots__ = ["x", "y", "z"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(
        self,
        x: float | None = ...,
        y: float | None = ...,
        z: float | None = ...,
    ) -> None: ...

class Vec3IntProto(_message.Message):
    __slots__ = ["x", "y", "z"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    z: int
    def __init__(
        self, x: int | None = ..., y: int | None = ..., z: int | None = ...
    ) -> None: ...
