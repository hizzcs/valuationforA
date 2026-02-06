"""A股估值平台 core package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("valuation-ashare")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
