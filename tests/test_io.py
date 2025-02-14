import os
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import pytest
from fsspec.implementations.local import LocalFileSystem
from google.oauth2.credentials import Credentials
from pandas.testing import assert_frame_equal
from pytest_mock import MockerFixture
from shapely import Point

from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)
testdata_folder = Path("tests", "data")


@pytest.fixture
def parquetfile_path() -> Iterator[str]:
    path = testdata_folder / "pointsw.parquet"
    yield str(path)
    path.unlink()


@pytest.fixture
def jsonfile_path() -> Iterator[str]:
    path = testdata_folder / "pointsw.json"
    yield str(path)
    path.unlink()


@pytest.fixture
def gpkgfile_path() -> Iterator[str]:
    path = testdata_folder / "pointsw.gpkg"
    yield str(path)
    path.unlink()


@pytest.fixture
def shpfile_path() -> Iterator[str]:
    path = testdata_folder / "pointsw.shp"
    yield str(path)
    path.unlink()
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        (testdata_folder / sidecar).unlink()


def test_read_parquet(mocker: MockerFixture) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    lestdataframe = read_dataframe(str(Path("tests", "data", "points.parquet")))
    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(mocker: MockerFixture) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    lestdataframe = read_dataframe(str(Path("tests", "data", "points.parquet")), bbox=[0.5, 1.5, 2.5, 3.5])
    assert_frame_equal(punkdataframe.iloc[:2], lestdataframe)


def test_write_parquet(mocker: MockerFixture, parquetfile_path: str) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    write_dataframe(punkdataframe, parquetfile_path)
    assert os.path.exists(parquetfile_path)


def test_roundtrip_parquet(mocker: MockerFixture, parquetfile_path: str) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    write_dataframe(punkdataframe, parquetfile_path)
    roundtrip = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, roundtrip)


def test_write_shp(mocker: MockerFixture, shpfile_path: str) -> None:
    mock_google_creds = mocker.Mock(spec=Credentials)
    mock_google_creds.token = None
    auth_client_mock = mocker.patch("dapla_geoio.io.AuthClient")
    auth_client_mock.fetch_google_credentials.return_value = mock_google_creds
    ensure_mock = mocker.patch("dapla_geoio.io._ensure_gs_vsi_prefix")
    ensure_mock.side_effect = lambda x: x

    write_dataframe(punkdataframe, shpfile_path)
    assert os.path.exists(shpfile_path)


def test_write_geojson(mocker: MockerFixture, jsonfile_path: str) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    write_dataframe(punkdataframe, jsonfile_path)
    assert os.path.exists(jsonfile_path)


def test_write_gpkg(mocker: MockerFixture, gpkgfile_path: str) -> None:
    mock_google_creds = mocker.Mock(spec=Credentials)
    mock_google_creds.token = None
    auth_client_mock = mocker.patch("dapla_geoio.io.AuthClient")
    auth_client_mock.fetch_google_credentials.return_value = mock_google_creds
    ensure_mock = mocker.patch("dapla_geoio.io._ensure_gs_vsi_prefix")
    ensure_mock.side_effect = lambda x: x

    write_dataframe(punkdataframe, gpkgfile_path)
    assert os.path.exists(gpkgfile_path)
