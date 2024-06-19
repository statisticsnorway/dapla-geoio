import os
import pathlib
from collections.abc import Iterator

import geopandas as gpd
import pytest
from fsspec.implementations.local import LocalFileSystem
from google.oauth2.credentials import Credentials
from pandas.testing import assert_frame_equal
from pytest_mock import MockerFixture
from shapely import Point

from dapla_geoio import read_geodataframe
from dapla_geoio import write_geodataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)
testdata_folder = pathlib.Path("tests", "data")


@pytest.fixture
def parquetfile_path() -> Iterator[str]:
    path = str(testdata_folder / "pointsw.parquet")
    yield path
    os.remove(path)


@pytest.fixture
def shpfile_path() -> Iterator[str]:
    path = str(testdata_folder / "pointsw.shp")
    yield path
    os.remove(path)
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        os.remove(testdata_folder / sidecar)


def test_read_parquet(mocker: MockerFixture) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    lestdataframe = read_geodataframe(
        str(pathlib.Path("tests", "data", "points.parquet"))
    )
    assert_frame_equal(punkdataframe, lestdataframe)


def test_write_parquet(mocker: MockerFixture, parquetfile_path: str) -> None:
    file_client_mock = mocker.patch("dapla_geoio.io.FileClient")
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()

    write_geodataframe(punkdataframe, parquetfile_path)
    assert os.path.exists(parquetfile_path)


def test_write_shp(mocker: MockerFixture, shpfile_path: str) -> None:
    mock_google_creds = mocker.Mock(spec=Credentials)
    mock_google_creds.token = None
    auth_client_mock = mocker.patch("dapla_geoio.io.AuthClient")
    auth_client_mock.fetch_google_credentials.return_value = mock_google_creds
    ensure_mock = mocker.patch("dapla_geoio.io._ensure_gs_vsi_prefix")
    ensure_mock.side_effect = lambda x: x

    write_geodataframe(punkdataframe, shpfile_path)
    assert os.path.exists(shpfile_path)
