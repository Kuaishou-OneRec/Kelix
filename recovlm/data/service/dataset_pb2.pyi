from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BatchSequenceInfo(_message.Message):
    __slots__ = ["client_id", "session_id", "image_token_len"]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_TOKEN_LEN_FIELD_NUMBER: _ClassVar[int]
    client_id: int
    session_id: int
    image_token_len: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, client_id: _Optional[int] = ..., session_id: _Optional[int] = ..., image_token_len: _Optional[_Iterable[int]] = ...) -> None: ...

class SelectedValue(_message.Message):
    __slots__ = ["client_id", "session_id", "selected"]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SELECTED_FIELD_NUMBER: _ClassVar[int]
    client_id: int
    session_id: int
    selected: int
    def __init__(self, client_id: _Optional[int] = ..., session_id: _Optional[int] = ..., selected: _Optional[int] = ...) -> None: ...
