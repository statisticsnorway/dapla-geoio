import os
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import pytest
from fsspec.implementations.local import LocalFileSystem
from google.oauth2.credentials import Credentials
from pandas.testing import assert_frame_equal
from pyarrow import fs
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
from shapely import Point

import dapla_geoio.io
from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)
testdata_folder = Path("tests", "data")


@pytest.fixture
def patch_gcs(monkeypatch: MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dapla_geoio.io.fs, "GcsFileSystem", fs.LocalFileSystem)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        dapla_geoio.io.FileClient, "get_gcs_file_system", lambda: LocalFileSystem()  # type: ignore[attr-defined]
    )
    yield


@pytest.fixture
def mock_autclient(mocker: MockerFixture) -> Iterator[None]:
    mock_google_creds = mocker.Mock(spec=Credentials)
    mock_google_creds.token = None
    auth_client_mock = mocker.patch("dapla_geoio.io.AuthClient")
    auth_client_mock.fetch_google_credentials.return_value = mock_google_creds
    yield


@pytest.fixture()
def mock_vsi_prefix(mocker: MockerFixture) -> Iterator[None]:
    ensure_mock = mocker.patch("dapla_geoio.io._ensure_gs_vsi_prefix")
    ensure_mock.side_effect = lambda x: x
    yield


@pytest.fixture
def parquetfile_path(patch_gcs: None) -> Iterator[Path]:
    path = testdata_folder / "pointsw.parquet"
    yield path
    path.unlink()


@pytest.fixture
def jsonfile_path(patch_gcs: None) -> Iterator[Path]:
    path = testdata_folder / "pointsw.json"
    yield path
    path.unlink()


@pytest.fixture
def gpkgfile_path(mock_autclient: None, mock_vsi_prefix: None) -> Iterator[Path]:
    path = testdata_folder / "pointsw.gpkg"
    yield path
    path.unlink()


@pytest.fixture
def shpfile_path(mock_autclient: None, mock_vsi_prefix: None) -> Iterator[Path]:
    path = testdata_folder / "pointsw.shp"
    yield path
    path.unlink()
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        (testdata_folder / sidecar).unlink()


def test_read_parquet(patch_gcs: None) -> None:
    path = Path("tests", "data", "points.parquet")
    lestdataframe = read_dataframe(str(path))
    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(patch_gcs: None) -> None:
    lestdataframe = read_dataframe(
        str(Path("tests", "data", "points.parquet")), bbox=[0.5, 1.5, 2.5, 3.5]
    )
    assert_frame_equal(punkdataframe.iloc[:2], lestdataframe)


def test_write_parquet(parquetfile_path: Path) -> None:
    write_dataframe(punkdataframe, str(parquetfile_path))
    assert parquetfile_path.exists()


def test_roundtrip_parquet(parquetfile_path: Path) -> None:
    write_dataframe(punkdataframe, str(parquetfile_path))
    roundtrip = read_dataframe(str(parquetfile_path))
    assert_frame_equal(punkdataframe, roundtrip)


def test_write_shp(shpfile_path: Path) -> None:
    write_dataframe(punkdataframe, str(shpfile_path))
    assert shpfile_path.exists()


def test_write_geojson(jsonfile_path: Path) -> None:
    write_dataframe(punkdataframe, str(jsonfile_path))
    assert jsonfile_path.exists()


def test_write_gpkg(gpkgfile_path: Path) -> None:
    write_dataframe(punkdataframe, str(gpkgfile_path))
    assert os.path.exists(gpkgfile_path)
