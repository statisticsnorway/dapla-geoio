import json
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pyarrow
import pyogrio
import shapely
from dapla import AuthClient
from dapla import FileClient
from geopandas.array import from_wkb
from pyarrow import parquet

_geometry_type_names = [
    "Point",
    "LineString",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]
_geometry_type_names.extend([geom_type + " Z" for geom_type in _geometry_type_names])


def set_gdal_auth() -> None:
    """Setter miljøvariabler for GDAL."""
    credentials = AuthClient.fetch_google_credentials()

    pyogrio.set_gdal_config_options(
        {
            "GDAL_HTTP_HEADERS": f"Authorization: Bearer {credentials.token}",
            "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": True,
        }
    )


def _ensure_gs_vsi_prefix(gcs_path: str) -> str:
    """Legger til prefixet "/vsigs/" til filbanen.

    GDAL har sitt eget virtuelle filsystem abstraksjons-skjema ulikt fsspec,
    som bruker prefixet /vsigs/ istedenfor gs:/.
    https://gdal.org/user/virtual_file_systems.html
    """
    if "/vsigs/" in gcs_path:
        return gcs_path

    if gcs_path.startswith(prefix := "gs:/"):
        return f"/vsigs/{gcs_path[len(prefix):]}"

    return f"/vsigs/{gcs_path}"


def _remove_prefix(gcs_path: str) -> str:
    """Fjerner både prefikset /vsigs/ og gs:/ fra filsti."""
    for prefix in ["gs:/", "/vsigs/"]:
        if gcs_path.startswith(prefix):
            return gcs_path[len(prefix) :]

    else:
        return gcs_path


def _get_geometry_types(series: gpd.GeoSeries) -> list[str]:
    arr_geometry_types = shapely.get_type_id(series)
    # ensure to include "... Z" for 3D geometries
    has_z = series.has_z
    arr_geometry_types[has_z] += 8

    geometry_types = arr_geometry_types.unique().tolist()
    # drop missing values (shapely.get_type_id returns -1 for those)
    if -1 in geometry_types:
        geometry_types.remove(-1)

    return sorted([_geometry_type_names[idx] for idx in geometry_types])


def read_geodataframe(
    gcs_path: str | Iterable[str],
    file_format: str | None = None,
    columns: Iterable[str] | None = None,
    # filters: list[tuple | list[tuple]] | pyarrow.compute.Expression | None = None,
    geometry_column: str | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Leser inn en fil som innholder geometri til en Geopandas geodataframe.

    Støtter geoparquetfiler med WKB kodet geometri og partisjonerte geoparquetfiler.
    Bruker pyogrio til å lese andre filformater.
    """
    if file_format is None:
        if isinstance(gcs_path, str):
            if gcs_path.endswith(".parquet"):
                file_format = "parquet"
        else:
            if all(path.endswith(".parquet") for path in gcs_path):
                file_format = "parquet"

    if file_format == "parquet":
        filesystem = FileClient.get_gcs_file_system()

        if isinstance(gcs_path, str):
            gcs_path = _remove_prefix(gcs_path)
        else:
            gcs_path = (_remove_prefix(file) for file in gcs_path)

        arrow_table = parquet.ParquetDataset(gcs_path, filesystem=filesystem).read(
            columns=columns, use_pandas_metadata=True
        )
        return _arrow_til_geopandas(arrow_table, geometry_column)

    else:
        if not isinstance(gcs_path, str):
            ValueError("Multiple paths are only supported for parquet format")

        set_gdal_auth()
        path = _ensure_gs_vsi_prefix(gcs_path)

        return pyogrio.read_dataframe(path, use_arrow=True, **kwargs)


def write_geodataframe(
    gdf: gpd.GeoDataFrame,
    gcs_path: str,
    file_format: str | None = None,
    **kwargs,
) -> None:
    """Skriver en Geopandas geodataframe til ei fil.

    Støtter  å skrive til geoparquetfiler med WKB kodet geometri, og
    bruker pyogrio til å lese andre filformater.
    """
    if file_format is None and gcs_path.endswith(".parquet"):
        file_format = "parquet"

    if file_format == "parquet":
        filesystem = FileClient.get_gcs_file_system()
        gcs_path = _remove_prefix(gcs_path)
        table = _geopandas_to_arrow(gdf)
        parquet.write_table(
            table,
            where=gcs_path,
            filesystem=filesystem,
            compression="snappy",
            coerce_timestamps="ms",
            **kwargs,
        )

    else:

        set_gdal_auth()
        path = _ensure_gs_vsi_prefix(gcs_path)

        pyogrio.write_dataframe(gdf, path, use_arrow=True, **kwargs)


def get_parquet_files_in_folder(folder: str) -> list[str]:
    """Lister opp parquetfiler i en "mappe" i en Google cloud bøtte.

    Nyttig hvis man har en partisjonert geoparquet fil.
    """
    file_paths = FileClient.ls(folder)
    return list(filter(lambda p: p.endswith(".parquet"), file_paths))


def _geopandas_to_arrow(gdf: gpd.GeoDataFrame) -> pyarrow.Table:
    """Kopiert og tilpasset privat funksjon fra https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py."""
    # create geo metadata before altering incoming data frame
    geo_metadata = _create_metadata(gdf)

    df = gdf.to_wkb()

    table = pyarrow.Table.from_pandas(df)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": json.dumps(geo_metadata).encode("utf-8")})

    return table.replace_schema_metadata(metadata)


def _arrow_til_geopandas(
    arrow_table: pyarrow.Table, geometry_column: str | None = None
) -> gpd.GeoDataFrame:
    """Kopiert og tilpasset privat funksjon fra https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py."""
    df = arrow_table.to_pandas()

    geometry_metadata = json.loads(arrow_table.schema.metadata[b"geo"].decode("utf-8"))

    geometry_columns = df.columns.intersection(geometry_metadata["columns"])

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use dapla.read_pandas() instead."""
        )

    geometry_column = (
        geometry_metadata["primary_column"] if not geometry_column else geometry_column
    )

    # Missing geometry likely indicates a subset of columns was read;
    # promote the first available geometry to the primary geometry.
    if geometry_column not in geometry_columns:
        raise ValueError(
            "Geometry column not in columns read from the Parquet/Feather file."
        )

    # Convert the WKB columns that are present back to geometry.
    for column in geometry_columns:
        column_metadata = geometry_metadata["columns"][column]
        if "crs" in column_metadata:
            crs = column_metadata["crs"]
        else:
            # per the GeoParquet spec, missing CRS is to be interpreted as
            # OGC:CRS84
            crs = "OGC:CRS84"

        if column_metadata["encoding"] == "WKB":
            df[column] = from_wkb(df[column].values, crs=crs)
        else:
            raise ValueError("Only WKB encoding of geometry is supported.")

    return gpd.GeoDataFrame(df, geometry=geometry_column)


def _create_metadata(gdf: gpd.GeoDataFrame) -> None:
    schema_version = "1.0.0"

    # Construct metadata for each geometry
    column_metadata = {}
    for col in gdf.columns[gdf.dtypes == "geometry"]:
        series = gdf[col]

        geometry_types = _get_geometry_types(series)
        geometry_types_name = "geometry_types"

        crs = series.crs.to_json_dict() if series.crs else None

        column_metadata[col] = {
            "encoding": "WKB",
            "crs": crs,
            geometry_types_name: geometry_types,
        }

        bbox = series.total_bounds.tolist()
        if np.isfinite(bbox).all():
            # don't add bbox with NaNs for empty / all-NA geometry column
            column_metadata[col]["bbox"] = bbox

        column_metadata[col]["covering"] = {
            "bbox": {
                "xmin": ["bbox", "xmin"],
                "ymin": ["bbox", "ymin"],
                "xmax": ["bbox", "xmax"],
                "ymax": ["bbox", "ymax"],
            },
        }

    return {
        "primary_column": gdf.geometry.name,
        "columns": column_metadata,
        "version": schema_version,
        "creator": {"library": "geopandas", "version": gpd.__version__},
    }
