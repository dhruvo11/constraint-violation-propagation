"""Registry of supported document types."""
from .certificate import Certificate
from .idcard import IDCard
from .invoice import Invoice
from .passport import Passport
from .receipt import Receipt
from .transcript import Transcript

_REGISTRY = {cls.name: cls() for cls in (Transcript, Invoice, Receipt, Certificate, Passport, IDCard)}


def get(name="transcript"):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown document type {name!r}; known: {', '.join(_REGISTRY)}") from None


def names():
    return list(_REGISTRY)
