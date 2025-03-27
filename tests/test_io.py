import os
from collections.abc import Iterator
from unittest.mock import Mock

import geopandas as gpd
import pyarrow.fs
import pytest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
from shapely import Point
from upath import UPath
from upath.implementations.cloud import GCSPath
from upath.implementations.local import PosixUPath
from upath.implementations.local import WindowsUPath

import dapla_geoio.io
from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)
local_upath = WindowsUPath if os.name == "nt" else PosixUPath


@pytest.fixture
def gcs_patch(monkeypatch: MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", value=local_upath)
    monkeypatch.setattr(dapla_geoio.io.pyarrow.fs, "GcsFileSystem", value=pyarrow.fs.LocalFileSystem)
    monkeypatch.setattr(dapla_geoio.io, "set_gdal_auth", lambda: None)
    monkeypatch.setattr(dapla_geoio.io, "_ensure_gs_vsi_prefix", lambda x: x.path)

    yield 

@pytest.fixture
def parquetfile_path(gcs_patch: None):
    path = local_upath("tests", "temp", "pointsw.parquet")
    yield path
    path.unlink()


@pytest.fixture
def jsonfile_path(gcs_patch: None):
    path = local_upath("tests", "temp", "pointsw.json")
    yield path
    path.unlink()


@pytest.fixture
def gpkgfile_path(gcs_patch: None):
    path = local_upath("tests", "temp", "pointsw.gpkg")
    yield path
    path.unlink()


@pytest.fixture
def shpfile_path(gcs_patch: None) :
    path = local_upath("tests", "temp", "pointsw.shp")
    yield path
    path.unlink()
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        local_upath("tests", "temp", sidecar).unlink()


def test_read_parquet(gcs_patch: None) -> None:
    parquetfile_path = local_upath("tests, data, points.parquet")
    lestdataframe = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(gcs_patch: None) -> None:
    parquetfile_path = local_upath("tests", "data", "points.parquet")
    lestdataframe = read_dataframe(
        parquetfile_path, bbox=[0.5, 1.5, 2.5, 3.5]
    )
    assert_frame_equal(punkdataframe.iloc[:2], lestdataframe)


def test_write_parquet(parquetfile_path: GCSPath) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    assert parquetfile_path.exists()


def test_roundtrip_parquet(parquetfile_path: GCSPath) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    roundtrip = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, roundtrip)


def test_write_shp(shpfile_path: GCSPath) -> None:
    write_dataframe(punkdataframe, shpfile_path)
    assert shpfile_path.exists()


def test_write_geojson(jsonfile_path: GCSPath) -> None:
    write_dataframe(punkdataframe, jsonfile_path)
    assert jsonfile_path.exists()


def test_write_gpkg(gpkgfile_path: GCSPath) -> None:
    write_dataframe(punkdataframe, gpkgfile_path)
    assert gpkgfile_path.exists()
