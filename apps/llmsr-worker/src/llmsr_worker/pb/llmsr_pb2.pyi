from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Program(_message.Message):
    __slots__ = ("skeleton", "score")
    SKELETON_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    skeleton: str
    score: float
    def __init__(self, skeleton: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class ProposeRequest(_message.Message):
    __slots__ = ("parents",)
    PARENTS_FIELD_NUMBER: _ClassVar[int]
    parents: _containers.RepeatedCompositeFieldContainer[Program]
    def __init__(self, parents: _Optional[_Iterable[_Union[Program, _Mapping]]] = ...) -> None: ...

class ProposeResponse(_message.Message):
    __slots__ = ("skeletons",)
    SKELETONS_FIELD_NUMBER: _ClassVar[int]
    skeletons: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, skeletons: _Optional[_Iterable[str]] = ...) -> None: ...

class ObserveRequest(_message.Message):
    __slots__ = ("skeleton",)
    SKELETON_FIELD_NUMBER: _ClassVar[int]
    skeleton: str
    def __init__(self, skeleton: _Optional[str] = ...) -> None: ...

class ObserveResponse(_message.Message):
    __slots__ = ("skeleton", "score")
    SKELETON_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    skeleton: str
    score: float
    def __init__(self, skeleton: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...
