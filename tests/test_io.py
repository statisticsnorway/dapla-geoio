import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Literal
from typing import cast

import geopandas as gpd
import pyarrow.fs
import pyogrio
import pytest
import requests
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from shapely import Point
from upath.implementations.cloud import GCSPath

import dapla_geoio.io
from dapla_geoio.io import read_dataframe
from dapla_geoio.io import write_dataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)


def stop_docker(container: str) -> None:
    cid = (
        subprocess.check_output(
            ("docker", "ps", "-a", "-q", "--filter", f"name={container}")
        )
        .strip()
        .decode()
    )
    if cid:
        subprocess.run(("docker", "rm", "-f", "-v", cid))


@pytest.fixture(scope="session")
def local_testdir() -> Path:
    return Path("tests", "data").absolute()


@pytest.fixture(scope="session")
def docker_gcs() -> (
    Iterator[Literal["https://storage.googleapis.com", "http://localhost:4443"]]
):
    url: Literal["https://storage.googleapis.com", "http://localhost:4443"]

    if os.environ.get("DAPLA_REGION") == "DAPLA_LAB":
        yield "https://storage.googleapis.com"
        return

    url = "http://localhost:4443"

    if shutil.which("docker") is None:
        pytest.skip("docker not installed")

    container = "gcsfs_test"
    cmd = (
        "docker",
        "run",
        "-d",
        "-p",
        "4443:4443",
        "--name",
        container,
        "fsouza/fake-gcs-server:latest",
        "--scheme",
        "http",
        "--external-url",
        url,
        "--public-host",
        url,
    )
    stop_docker(container)
    subprocess.run(cmd)
    retries = 15
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url + "/storage/v1/b", timeout=10)
            if r.ok:
                yield url
                break
        except Exception as e:
            if attempt == retries:
                raise SystemError from e
            time.sleep(1)
    stop_docker(container)


@pytest.fixture(scope="session")
def gcs_fixture(docker_gcs: str, local_testdir: Path) -> Iterator[GCSPath]:
    bucket_path: GCSPath
    if os.environ.get("DAPLA_REGION") == "DAPLA_LAB":
        bucket_name = "ssb-dapla-felles-data-produkt-test"
        bucket_path = cast(  # type: ignore [redundant-cast]
            GCSPath,
            GCSPath(
                bucket_name,
                "dapla_geo_io_tests",
                protocol="gs",
                endpoint_url=docker_gcs,
            ),
        )

    else:
        bucket_name = "test_bucket"
        bucket_path = cast(  # type: ignore [redundant-cast]
            GCSPath,
            GCSPath(bucket_name, protocol="gs", endpoint_url=docker_gcs, token="anon"),
        )

    fs = bucket_path.fs

    if bucket_path.exists():
        for key in bucket_path.iterdir():  # type: ignore [no-untyped-call]
            if key.is_file():
                key.unlink()
    else:
        fs.mkdir(str(bucket_path))

    for source_path in local_testdir.iterdir():
        target_path = bucket_path / source_path.relative_to(local_testdir)

        if source_path.is_file():
            fs.upload(str(source_path), str(target_path))

    fs.invalidate_cache()

    yield bucket_path


@pytest.fixture
def gdal_patch(docker_gcs: str) -> Iterator[None]:
    if docker_gcs != "https://storage.googleapis.com":
        # fsouza/fake-gcs-server støtter ikke GCS XML-apiet, som Gdal er avhengig av.
        # Se https://github.com/fsouza/fake-gcs-server/issues/331
        # Et Docker image med nødvendige endringer finnes på tustvold/fake-gcs-server,
        # men denne inneholder nye bugs for Json-apiet.
        pytest.skip("Kan kun teste mot ekte GCS-tjeneste")
    else:
        anonym_innloging = False

    pyogrio.set_gdal_config_options(
        {
            "CPL_GS_ENDPOINT": docker_gcs + "/",
            "GS_NO_SIGN_REQUEST": anonym_innloging,
            "CPL_CURL_VERBOSE": True,
        }
    )
    yield


@pytest.fixture
def pyarrrow_patch(monkeypatch: MonkeyPatch, docker_gcs: str) -> Iterator[None]:
    if docker_gcs != "https://storage.googleapis.com":
        test_gcs_file_system = partial(
            pyarrow.fs.GcsFileSystem,
            anonymous=True,
            endpoint_override=docker_gcs.removeprefix("http://"),
        )
        monkeypatch.setattr(
            dapla_geoio.io.pyarrow.fs, "GcsFileSystem", value=test_gcs_file_system  # type: ignore [attr-defined]
        )
    yield


@pytest.fixture
def parquetfile_path(gcs_fixture: GCSPath, pyarrrow_patch: None) -> Iterator[GCSPath]:
    path = gcs_fixture / "temp" / "pointsw.parquet"
    yield path
    path.unlink()


@pytest.fixture
def jsonfile_path(gcs_fixture: GCSPath) -> Iterator[GCSPath]:
    path = gcs_fixture / "temp" / "pointsw.json"
    yield path
    path.unlink()


@pytest.fixture
def gpkgfile_path(gcs_fixture: GCSPath, gdal_patch: None) -> Iterator[GCSPath]:
    path = gcs_fixture / "temp" / "pointsw.gpkg"
    yield path
    path.unlink()


@pytest.fixture
def shpfile_path(gcs_fixture: GCSPath, gdal_patch: None) -> Iterator[GCSPath]:
    path = gcs_fixture / "temp" / "pointsw.shp"
    yield path
    path.unlink()
    for sidecar in ("pointsw.cpg", "pointsw.dbf", "pointsw.shx"):
        (gcs_fixture / "temp" / sidecar).unlink()


def test_read_parquet(gcs_fixture: GCSPath, pyarrrow_patch: None) -> None:
    parquetfile_path = gcs_fixture / "points.parquet"
    lestdataframe = read_dataframe(parquetfile_path)
    assert_frame_equal(punkdataframe, lestdataframe)


def test_read_parquet_bbox(gcs_fixture: GCSPath, pyarrrow_patch: None) -> None:
    parquetfile_path = gcs_fixture / "points.parquet"
    lestdataframe = read_dataframe(parquetfile_path, bbox=[0.5, 1.5, 2.5, 3.5])
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
