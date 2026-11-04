"""Registry of supported document types."""
from .transcript import Transcript

_REGISTRY = {cls.name: cls() for cls in (Transcript,)}


def get(name="transcript"):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown document type {name!r}; known: {', '.join(_REGISTRY)}") from None


def names():
    return list(_REGISTRY)
