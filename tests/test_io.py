import shutil
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import pytest
from fsspec.implementations.local import LocalFileSystem
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
from shapely import Point
import pyarrow.fs

import dapla_geoio.io
from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)
testdata_folder = Path("tests", "temp")


@pytest.fixture
def geoarrow_point_parquetfile(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", value=Path)
    monkeypatch.setattr(dapla_geoio.io.pyarrow.fs, "GcsFileSystem", value=pyarrow.fs.LocalFileSystem)

    path = testdata_folder / "points_geoarrow.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    punkdataframe.to_parquet(path, geometry_encoding="geoarrow")
    yield path
    shutil.rmtree(path.parent)


@pytest.fixture
def parquetfile_path(monkeypatch: MonkeyPatch, mocker: MockerFixture) -> Iterator[Path]:
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", Path)
    monkeypatch.setattr(dapla_geoio.io.pyarrow.fs, "GcsFileSystem", value=pyarrow.fs.LocalFileSystem)

    path = testdata_folder / "points.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path.parent)

@pytest.fixture
def jsonfile_path(monkeypatch: MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", Path)

    path = testdata_folder / "points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path.parent)

@pytest.fixture
def gpkgfile_path(monkeypatch: MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(dapla_geoio.io, "set_gdal_auth", lambda: None)
    monkeypatch.setattr(dapla_geoio.io, "_ensure_gs_vsi_prefix", lambda x: str(x))
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", Path)

    path = testdata_folder / "points.gpkg"
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path.parent)

@pytest.fixture
def shpfile_path(monkeypatch: MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(dapla_geoio.io, "set_gdal_auth", lambda: None)
    monkeypatch.setattr(dapla_geoio.io, "_ensure_gs_vsi_prefix", lambda x: str(x))
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", Path)

    path = testdata_folder / "points.shp"
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path.parent)


def test_read_parquet(geoarrow_point_parquetfile: Path) -> None:
    lestdataframe = read_dataframe(geoarrow_point_parquetfile)

    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(geoarrow_point_parquetfile: Path) -> None:  
    lestdataframe = read_dataframe(
        geoarrow_point_parquetfile, bbox=[0.5, 1.5, 2.5, 3.5]
    )
    assert_frame_equal(punkdataframe.iloc[:2], lestdataframe)


def test_write_parquet(parquetfile_path: Path) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    assert parquetfile_path.exists()


def test_roundtrip_parquet(parquetfile_path: Path) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    roundtrip = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, roundtrip)


def test_write_shp(shpfile_path: Path) -> None:
    write_dataframe(punkdataframe, shpfile_path)
    assert shpfile_path.exists()


def test_write_geojson(jsonfile_path: Path) -> None:
    write_dataframe(punkdataframe, jsonfile_path)
    assert jsonfile_path.exists()


def test_write_gpkg(gpkgfile_path: Path) -> None:
    write_dataframe(punkdataframe, gpkgfile_path)
    assert gpkgfile_path.exists()
