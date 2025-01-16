import io
import json
import shutil
import sys
from collections.abc import Iterable
from typing import Any
from typing import Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Required
    from typing import TypedDict
else:
    from strenum import StrEnum
    from typing_extensions import Required
    from typing_extensions import TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import pyogrio
import pyogrio._err
import pyogrio.errors
import shapely
from dapla import AuthClient
from dapla import FileClient
from geopandas.array import from_shapely
from geopandas.array import from_wkb
from geopandas.io._geoarrow import construct_geometry_array
from geopandas.io._geoarrow import construct_shapely_array
from geopandas.io._geoarrow import construct_wkb_array
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


def homogen_geometri(geoserie: gpd.GeoSeries) -> bool | np.bool_:
    """Sjekker at alle elementer i serien har lik geometritype og ikke er av typen GeometryCollection."""
    notnamaske = geoserie.notna()
    if not notnamaske.any():
        raise RuntimeError(
            f"Geometrikolonnen {geoserie.name} innholder kun tomme rader"
        )

    notna = geoserie[notnamaske]
    første: shapely.geometry.base.BaseGeometry = notna.iat[0]
    return (
        første.geom_type != "GeometryCollection"
        and (notna.geom_type == første.geom_type).all()
    )


class FileFormat(StrEnum):
    """En samling filformater som er garantert støttet."""

    PARQUET = "parquet"
    GEOPACKAGE = "GPKG"
    GEOJSON = "GeoJSON"
    FILEGDB = "OpenFileGDB"
    FLATGEOBUFFER = "FlatGeobuf"
    SHAPEFILE = "ESRI Shapefile"


filextension2format = {
    "parquet": FileFormat.PARQUET,
    "gpkg": FileFormat.GEOPACKAGE,
    "fgb": FileFormat.FLATGEOBUFFER,
    "json": FileFormat.GEOJSON,
    "geojson": FileFormat.GEOJSON,
    "shp": FileFormat.SHAPEFILE,
    "gdb": FileFormat.FILEGDB,
}


class _GeoParquetColumnMetadata(TypedDict, total=False):
    encoding: Required[str]
    geometry_types: Required[list[str]]
    crs: str | dict[str, Any] | None
    orientation: Literal["counterclockwise"]
    edges: Literal["planar", "spherical"]
    bbox: list[float]
    epoch: float
    covering: dict[str, Any]


class _GeoParquetMetadata(TypedDict):
    version: str
    primary_column: str
    columns: dict[str, _GeoParquetColumnMetadata]


def set_gdal_auth() -> None:
    """Setter miljøvariabler for GDAL."""
    options: dict[str, str | bool] = {"CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": True}

    credentials = AuthClient.fetch_google_credentials()
    if credentials.token:
        options["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {credentials.token}"

    pyogrio.set_gdal_config_options(options)


def _ensure_gs_vsi_prefix(gcs_path: str) -> str:
    """Legger til prefixet "/vsigs/" til filbanen.

    GDAL har sitt eget virtuelle filsystem abstraksjons-skjema ulikt fsspec,
    som bruker prefixet /vsigs/ istedenfor gs://.
    https://gdal.org/user/virtual_file_systems.html
    """
    if "/vsigs/" in gcs_path:
        return gcs_path

    if gcs_path.startswith(prefix := "gs://"):
        return f"/vsigs/{gcs_path[len(prefix):]}"

    return f"/vsigs/{gcs_path}"


def _remove_prefix(gcs_path: str) -> str:
    """Fjerner både prefikset /vsigs/ og gs:// fra filsti."""
    for prefix in ["gs://", "/vsigs/"]:
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


def read_dataframe(
    gcs_path: str | Iterable[str],
    file_format: FileFormat | None = None,
    columns: list[str] | None = None,
    # filters: list[tuple | list[tuple]] | pyarrow.compute.Expression | None = None,
    geometry_column: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Leser inn en fil som innholder geometri til en Geopandas geodataframe.

    Støtter geoparquetfiler med WKB kodet geometri og partisjonerte geoparquetfiler.
    Bruker pyogrio til å lese andre filformater.
    """
    if file_format is None:
        if isinstance(gcs_path, str):
            extension = gcs_path.rpartition(".")[-1]
            file_format = filextension2format.get(extension)

        elif all(path.endswith(".parquet") for path in gcs_path):
            file_format = FileFormat.PARQUET

    if file_format == FileFormat.PARQUET:
        filesystem = FileClient.get_gcs_file_system()

        if isinstance(gcs_path, str):
            gcs_path = _remove_prefix(gcs_path)
            return gpd.read_parquet(
                gcs_path, columns=columns, filesystem=filesystem, **kwargs
            )
        else:
            gcs_path = [_remove_prefix(file) for file in gcs_path]

            arrow_table = parquet.ParquetDataset(gcs_path, filesystem=filesystem).read(  # type: ignore[arg-type]
                columns=columns, use_pandas_metadata=True
            )
            return _arrow_til_geopandas(arrow_table, geometry_column)

    else:
        if not isinstance(gcs_path, str):
            raise ValueError("Multiple paths are only supported for parquet format")

        set_gdal_auth()
        path = _ensure_gs_vsi_prefix(gcs_path)

        try:
            return pyogrio.read_dataframe(
                path,
                columns=columns,
                driver=(str(file_format) if file_format else None),
                use_arrow=True,
                **kwargs,
            )
        except (
            pyogrio.errors.DataSourceError,
            pyogrio.errors.DataLayerError,
            pyogrio._err.CPLE_AppDefinedError,
        ) as e:
            # Reserve metode.
            # Fungerer ikke med formater som må lese fra flere filer.
            if file_format in {FileFormat.SHAPEFILE, FileFormat.FILEGDB}:
                raise e

            gcs_path = _remove_prefix(gcs_path)
            filesystem = FileClient.get_gcs_file_system()
            with filesystem.open(gcs_path, "rb") as buffer:
                return pyogrio.read_dataframe(
                    buffer,
                    columns=columns,
                    driver=(str(file_format) if file_format else None),
                    use_arrow=True,
                    **kwargs,
                )


def write_dataframe(
    gdf: gpd.GeoDataFrame,
    gcs_path: str,
    file_format: FileFormat | None = None,
    **kwargs: Any,
) -> None:
    """Skriver en Geopandas geodataframe til ei fil.

    Støtter  å skrive til geoparquetfiler med WKB kodet geometri, og
    bruker pyogrio til å lese andre filformater.
    """
    if file_format is None:
        extension = gcs_path.rpartition(".")[-1]
        file_format = filextension2format.get(extension)

    if file_format == FileFormat.PARQUET:
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

    elif file_format == FileFormat.GEOJSON:
        if (gdf.dtypes == "geometry").sum() != 1:
            raise ValueError("Geojson-formatet støtter kun en geometri kolonne")

        gcs_path = _remove_prefix(gcs_path)

        feature_collection = gdf.to_json(ensure_ascii=False)

        filesystem = FileClient.get_gcs_file_system()
        with filesystem.open(gcs_path, "w", encoding="utf-8") as file:
            file.write(feature_collection)

    else:
        set_gdal_auth()
        path = _ensure_gs_vsi_prefix(gcs_path)

        try:
            pyogrio.write_dataframe(
                gdf,
                path,
                driver=(str(file_format) if file_format else None),
                use_arrow=True,
                **kwargs,
            )

        except (
            pyogrio.errors.DataSourceError,
            pyogrio.errors.DataLayerError,
            pyogrio._err.CPLE_AppDefinedError,
        ) as e:
            # Reserve metode, først skrive til BytesIO, så til fil.
            # Fungerer ikke med formater som må skrive til flere filer.
            if file_format in {FileFormat.SHAPEFILE, FileFormat.FILEGDB, None}:
                raise e

            gcs_path = _remove_prefix(gcs_path)

            with io.BytesIO() as buffer:
                pyogrio.write_dataframe(
                    gdf,
                    buffer,
                    driver=str(file_format),
                    use_arrow=True,
                    **kwargs,
                )

                filesystem = FileClient.get_gcs_file_system()
                with filesystem.open(gcs_path, "wb") as file:
                    shutil.copyfileobj(buffer, file)


def get_parquet_files_in_folder(folder: str) -> list[str]:
    """Lister opp parquetfiler i en "mappe" i en Google cloud bøtte.

    Nyttig hvis man har en partisjonert geoparquet fil.
    """
    file_paths = FileClient.ls(folder)
    return list(filter(lambda p: p.endswith(".parquet"), file_paths))


def _geopandas_to_arrow(gdf: gpd.GeoDataFrame) -> pyarrow.Table:
    """Kopiert og tilpasset privat funksjon fra https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py."""
    mask = gdf.dtypes == "geometry"
    geometry_columns = gdf.columns[mask]
    geometry_indices = np.asarray(mask).nonzero()[0]

    df_attr = pd.DataFrame(gdf.copy(deep=False))
    for geo_column in geometry_columns:
        df_attr[geo_column] = None

    table = pyarrow.Table.from_pandas(df_attr)

    geometry_encoding_dict = {}

    for i, geo_column in zip(geometry_indices, geometry_columns, strict=False):
        geo_series = gdf[geo_column]
        # Bruker geoarrow koding på alle kolonner hvor det er mulig, og WKB ellers.
        encoding = "geoarrow" if homogen_geometri(geo_series) else "WKB"

        if encoding == "geoarrow":
            field, geom_arr = construct_geometry_array(
                np.array(geo_series.array),
                field_name=geo_column,
                crs=gdf.crs,
            )

            geometry_encoding_dict[geo_column] = (
                field.metadata[b"ARROW:extension:name"]
                .decode()
                .removeprefix("geoarrow.")
            )

        elif encoding == "WKB":
            field, geom_arr = construct_wkb_array(
                np.asarray(geo_series.array), field_name=geo_column, crs=gdf.crs
            )
            geometry_encoding_dict[geo_column] = encoding

        table = table.set_column(i, field, geom_arr)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    geo_metadata = _create_metadata(gdf, geometry_encoding_dict)
    metadata = table.schema.metadata if table.schema.metadata else {}
    metadata.update({b"geo": json.dumps(geo_metadata).encode("utf-8")})

    return table.replace_schema_metadata(metadata)


def _arrow_til_geopandas(
    arrow_table: pyarrow.Table, geometry_column: str | None = None
) -> gpd.GeoDataFrame:
    """Kopiert og tilpasset privat funksjon fra https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py."""
    geometry_metadata = _get_geometry_metadata(arrow_table.schema)

    geometry_columns = [
        col for col in geometry_metadata["columns"] if col in arrow_table.column_names
    ]
    result_column_names = list(arrow_table.slice(0, 0).to_pandas().columns)
    geometry_columns.sort(key=result_column_names.index)

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
    if geometry_column not in geometry_columns:
        raise ValueError(
            "Geometry column not in columns read from the Parquet/Feather file."
        )

    table_attr = arrow_table.drop(geometry_columns)
    df = table_attr.to_pandas()

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
            geom_arr = from_wkb(np.array(arrow_table[column]), crs=crs)
        else:
            geom_arr = from_shapely(
                construct_shapely_array(
                    arrow_table[column].combine_chunks(),
                    "geoarrow." + column_metadata["encoding"],
                ),
                crs=crs,
            )

        df.insert(result_column_names.index(column), column, geom_arr)

    return gpd.GeoDataFrame(df, geometry=geometry_column)


def _get_geometry_metadata(schema: pyarrow.Schema) -> _GeoParquetMetadata:
    metadata = schema.metadata if schema.metadata else {}
    try:
        geo_metadata_bytes = metadata[b"geo"]

    except KeyError:
        raise ValueError(
            """No geometry metadata is present.
            To read a table without geometry, use dapla.read_pandas() instead"""
        ) from None

    return json.loads(geo_metadata_bytes.decode("utf-8"))  # type: ignore[no-any-return]


def _create_metadata(
    gdf: gpd.GeoDataFrame, geometry_encoding: dict[str, str]
) -> _GeoParquetMetadata:
    schema_version = "1.0.0"

    # Construct metadata for each geometry
    columns_metadata: dict[str, _GeoParquetColumnMetadata] = {}
    for col in gdf.columns[gdf.dtypes == "geometry"]:
        series = gdf[col]

        geometry_types = _get_geometry_types(series)

        crs = series.crs.to_json_dict() if series.crs else None

        column_metadata: _GeoParquetColumnMetadata = {
            "encoding": geometry_encoding[col],
            "crs": crs,
            "geometry_types": geometry_types,
        }

        bbox = series.total_bounds.tolist()
        if np.isfinite(bbox).all():
            # don't add bbox with NaNs for empty / all-NA geometry column
            column_metadata["bbox"] = bbox

        columns_metadata[col] = column_metadata

    return {
        "primary_column": gdf.geometry.name,
        "columns": columns_metadata,
        "version": schema_version,
    }
