"""Dapla GeoIO."""

from .io import read_dataframe as read_pandas
from .io import write_dataframe as write_pandas

__all__ = ["read_pandas", "write_pandas"]
