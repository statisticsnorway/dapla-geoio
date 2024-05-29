from unittest import mock

import geopandas as gpd
from fsspec.implementations.local import LocalFileSystem
from shapely import Point

from dapla_geoio import write_geodataframe

punktserie = gpd.GeoSeries([Point((1, 2)), Point((2, 3)), Point((3, 4))])
punkdataframe = gpd.GeoDataFrame({"poi": ("a", "b", "c")}, geometry=punktserie)


@mock.patch("dapla.pandas.FileClient")
def test_write_parquet(file_client_mock: mock.Mock) -> None:
    file_client_mock.get_gcs_file_system.return_value = LocalFileSystem()
    write_geodataframe(punkdataframe, "tests/data/points.parquet")
