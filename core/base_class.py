from dataclasses import dataclass
import types
import typing
from .logger import Logger


@dataclass
class BaseClass:
    """
    Parent class of all classes that performs validation of every type
    """

    def __post_init__(self):
        try:
            hints = typing.get_type_hints(type(self))
        except Exception:
            hints = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            field_type = hints.get(field_name, field_def.type)
            actual_type = self._resolve_type(field_type)
            if actual_type is None:
                continue
            actual_value = getattr(self, field_name)

            if not isinstance(actual_value, actual_type):
                raise ValueError(
                    f"\t{field_name}: '{type(actual_value)}' instead of '{field_type}'"
                )

    @staticmethod
    def _resolve_type(field_type):
        """
        Convert a type hint into something usable by isinstance, or None if the
        field cannot be checked (typing.Any, ClassVar, Literal, strings, ...).
        """
        if field_type is typing.Any or isinstance(field_type, (str, typing.TypeVar)):
            return None
        origin = typing.get_origin(field_type)
        if origin is typing.Union or origin is types.UnionType:
            resolved = []
            for arg in typing.get_args(field_type):
                arg_type = BaseClass._resolve_type(arg)
                if arg_type is None:
                    # an unchecked member makes the whole union unchecked
                    return None
                resolved.append(arg_type)
            return tuple(resolved)
        if origin is not None:
            return origin if isinstance(origin, type) else None
        if field_type is None:
            return type(None)
        return field_type if isinstance(field_type, type) else None
