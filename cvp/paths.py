"""Where everything lives on disk. Each document type gets its own data, feature and
result folders; trained models are shared under models/."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
REPORT = os.path.join(BASE, "report")


def data_dir(doctype):
    return os.path.join(BASE, "data", doctype)


def feature_dir(doctype):
    return os.path.join(BASE, "features", doctype)


def results_dir(doctype):
    return os.path.join(BASE, "results", doctype)


def real_dir(dataset):
    """A downloaded public dataset (never under version control)."""
    return os.path.join(BASE, "data", "real", dataset)


PRIVATE = os.path.join(BASE, "private")  # the user's own documents (never under version control)
