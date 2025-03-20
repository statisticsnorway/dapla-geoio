from collections.abc import Iterator

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from shapely import Point
import pyarrow.fs

import dapla_geoio.io
from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe
from upath import UPath


punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)


testdata_folder = UPath("tests", "temp")


@pytest.fixture
def patch_gcs(monkeypatch: MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dapla_geoio.io, "GCSPath", value=UPath)
    monkeypatch.setattr(dapla_geoio.io.pyarrow.fs, "GcsFileSystem", value=pyarrow.fs.LocalFileSystem)
    monkeypatch.setattr(dapla_geoio.io, "set_gdal_auth", lambda: None)
    monkeypatch.setattr(dapla_geoio.io, "_ensure_gs_vsi_prefix", lambda x: x.path)

    yield

@pytest.fixture
def parquetfile_path(patch_gcs: None) -> Iterator[UPath]:
    path = testdata_folder / "pointsw.parquet"
    yield path
    path.unlink()


@pytest.fixture
def jsonfile_path(patch_gcs: None) -> Iterator[UPath]:
    path = testdata_folder / "pointsw.json"
    yield path
    path.unlink()


@pytest.fixture
def gpkgfile_path(patch_gcs: None) -> Iterator[UPath]:
    path = testdata_folder / "pointsw.gpkg"
    yield path
    path.unlink()


@pytest.fixture
def shpfile_path(patch_gcs: None) -> Iterator[UPath]:
    path = testdata_folder / "pointsw.shp"
    yield path
    path.unlink()
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        (testdata_folder / sidecar).unlink()


def test_read_parquet(patch_gcs: None) -> None:
    parquetfile_path = UPath("tests", "data", "points.parquet")
    lestdataframe = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(patch_gcs: None) -> None:
    parquetfile_path = UPath("tests", "data", "points.parquet")
    lestdataframe = read_dataframe(
        parquetfile_path, bbox=[0.5, 1.5, 2.5, 3.5]
    )
    assert_frame_equal(punkdataframe.iloc[:2], lestdataframe)


def test_write_parquet(parquetfile_path: UPath) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    assert parquetfile_path.exists()


def test_roundtrip_parquet(parquetfile_path: UPath) -> None:
    write_dataframe(punkdataframe, parquetfile_path)
    roundtrip = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, roundtrip)


def test_write_shp(shpfile_path: UPath) -> None:
    write_dataframe(punkdataframe, shpfile_path)
    assert shpfile_path.exists()


def test_write_geojson(jsonfile_path: UPath) -> None:
    write_dataframe(punkdataframe, jsonfile_path)
    assert jsonfile_path.exists()


def test_write_gpkg(gpkgfile_path: UPath) -> None:
    write_dataframe(punkdataframe, gpkgfile_path)
    assert gpkgfile_path.exists()
