from __future__ import annotations

import io
import json
import shutil
from collections.abc import Iterable
from collections.abc import Iterator
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import pyarrow.fs
import pyogrio
import pyogrio._err
import pyogrio.errors
import shapely
from geopandas.array import from_shapely
from geopandas.array import from_wkb
from geopandas.io._geoarrow import construct_geometry_array
from geopandas.io._geoarrow import construct_shapely_array
from geopandas.io._geoarrow import construct_wkb_array
from pyarrow import parquet
from upath.implementations.cloud import GCSPath

from .geoparquet import BoundingBox
from .geoparquet import GeoParquetDataset
from .geoparquet import _GeoParquetColumnMetadata
from .geoparquet import _GeoParquetMetadata

if TYPE_CHECKING:
    from pyarrow._stubs_typing import FilterTuple


_GEOMETRY_TYPE_NAMES = [
    "Point",
    "LineString",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]
_GEOMETRY_TYPE_NAMES.extend([geom_type + " Z" for geom_type in _GEOMETRY_TYPE_NAMES])


class FileFormat(StrEnum):
    """En samling filformater som er garantert støttet."""

    PARQUET = "parquet"
    GEOPACKAGE = "GPKG"
    GEOJSON = "GeoJSON"
    FILEGDB = "OpenFileGDB"
    FLATGEOBUFFER = "FlatGeobuf"
    SHAPEFILE = "ESRI Shapefile"


_FILE_EXTENSTION2FORMAT = {
    ".parquet": FileFormat.PARQUET,
    ".gpkg": FileFormat.GEOPACKAGE,
    ".fgb": FileFormat.FLATGEOBUFFER,
    ".json": FileFormat.GEOJSON,
    ".geojson": FileFormat.GEOJSON,
    ".shp": FileFormat.SHAPEFILE,
    ".gdb": FileFormat.FILEGDB,
}


def homogen_geometri(geoserie: gpd.GeoSeries) -> bool | np.bool_:
    """Sjekker at alle elementer i serien har lik geometritype og ikke er av typen GeometryCollection."""
    notnamaske = geoserie.notna()
    if not notnamaske.any():
        raise RuntimeError(
            f"The geometry column {geoserie.name} contains only empty rows"
        )

    notna = geoserie[notnamaske]
    første: shapely.geometry.base.BaseGeometry = notna.iat[0]
    return (
        første.geom_type != "GeometryCollection"
        and (notna.geom_type == første.geom_type).all()
    )


def set_gdal_auth() -> None:
    """Setter miljøvariabler for GDAL."""
    options: dict[str, str | bool] = {
        "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": True,
    }

    pyogrio.set_gdal_config_options(options)


def _ensure_gs_path(
    path_or_paths: str | GCSPath | Iterable[str] | Iterable[GCSPath],
) -> list[GCSPath]:
    if isinstance(path_or_paths, GCSPath):
        return [path_or_paths]

    elif isinstance(path_or_paths, str):
        return [cast(GCSPath, GCSPath(path_or_paths, protocol="gs"))]  # type: ignore [redundant-cast]

    elif any(isinstance(path, GCSPath) for path in path_or_paths):
        return cast(list[GCSPath], list(path_or_paths))

    else:
        return [cast(GCSPath, GCSPath(path, protocol="gs")) for path in path_or_paths]  # type: ignore [redundant-cast]


def _ensure_gs_vsi_prefix(gcs_path: GCSPath) -> str:
    """Legger til prefixet "/vsigs/" til filbanen.

    GDAL har sitt eget virtuelle fil-abstraksjons-system ulikt fsspec,
    som bruker prefixet /vsigs/ istedenfor gs://.
    https://gdal.org/user/virtual_file_systems.html
    """
    return f"/vsigs/{gcs_path.path.removeprefix(f'{gcs_path.protocol}://')}"


def _get_geometry_types(series: gpd.GeoSeries) -> list[str]:
    arr_geometry_types = shapely.get_type_id(series)
    # ensure to include "... Z" for 3D geometries
    has_z = series.has_z
    arr_geometry_types[has_z] += 8

    geometry_types = arr_geometry_types.unique().tolist()
    # drop missing values (shapely.get_type_id returns -1 for those)
    if -1 in geometry_types:
        geometry_types.remove(-1)

    return sorted([_GEOMETRY_TYPE_NAMES[idx] for idx in geometry_types])


def read_dataframe(
    path_or_paths: str | GCSPath | Iterable[str] | Iterable[GCSPath],
    file_format: FileFormat | None = None,
    columns: list[str] | None = None,
    bbox: Iterable[float] | BoundingBox | None = None,
    filters: list[FilterTuple | list[FilterTuple]] | ds.Expression | None = None,
    geometry_column: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Leser inn en fil som innholder geometri til en Geopandas geodataframe.

    Støtter geoparquetfiler med WKB kodet geometri og partisjonerte geoparquetfiler.
    Bruker pyogrio til å lese andre filformater.
    """
    gcs_paths = _ensure_gs_path(path_or_paths)

    if file_format is None:
        if all(path.suffix == "parquet" for path in gcs_paths):
            file_format = FileFormat.PARQUET

        else:
            extension = gcs_paths[0].suffix
            file_format = _FILE_EXTENSTION2FORMAT.get(extension)

    if file_format == FileFormat.PARQUET:
        dataset = GeoParquetDataset(
            gcs_paths,
            geometry_column=geometry_column,
            filters=filters,
            bbox=bbox,
            **kwargs,
        )

        arrow_table = dataset.read(columns=columns)

        return _arrow_til_geopandas(
            arrow_table, dataset.geometry_metadata, geometry_column
        )

    else:
        if len(gcs_paths) > 1:
            raise ValueError("Multiple paths are only supported for parquet format")

        set_gdal_auth()
        gcs_path = gcs_paths[0]
        vsi_path = _ensure_gs_vsi_prefix(gcs_path)

        try:
            return pyogrio.read_dataframe(
                vsi_path,
                bbox=bbox,
                columns=columns,
                driver=(file_format.value if file_format else None),
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

            with gcs_path.open("rb") as buffer:
                return pyogrio.read_dataframe(
                    buffer,
                    bbox=bbox,
                    columns=columns,
                    driver=(file_format.value if file_format else None),
                    use_arrow=True,
                    **kwargs,
                )


def write_dataframe(
    gdf: gpd.GeoDataFrame,
    path: str | GCSPath,
    file_format: FileFormat | None = None,
    **kwargs: Any,
) -> None:
    """Skriver en Geopandas geodataframe til ei fil.

    Støtter  å skrive til geoparquetfiler med WKB kodet geometri, og
    bruker pyogrio til å lese andre filformater.
    """
    gsc_path = _ensure_gs_path(path)[0]

    if file_format is None:
        extension = gsc_path.suffix
        file_format = _FILE_EXTENSTION2FORMAT.get(extension)

    if file_format == FileFormat.PARQUET:
        filesystem = pyarrow.fs.GcsFileSystem()
        table = _geopandas_to_arrow(gdf)
        parquet.write_table(
            table,
            where=gsc_path.path,
            filesystem=filesystem,
            compression="snappy",
            coerce_timestamps="ms",
            **kwargs,
        )

    elif file_format == FileFormat.GEOJSON:
        if (gdf.dtypes == "geometry").sum() != 1:
            raise ValueError("The Geojson-format supports only on geometry column.")

        feature_collection = gdf.to_json(ensure_ascii=False)

        with gsc_path.open("w", encoding="utf-8") as file:
            file.write(feature_collection)

    else:
        set_gdal_auth()
        vsi_path = _ensure_gs_vsi_prefix(gsc_path)

        try:
            pyogrio.write_dataframe(
                gdf,
                vsi_path,
                driver=(file_format.value if file_format else None),
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

            with io.BytesIO() as buffer:
                pyogrio.write_dataframe(
                    gdf,
                    buffer,
                    driver=(file_format.value if file_format else None),
                    use_arrow=True,
                    **kwargs,
                )

                with gsc_path.open("wb") as file:
                    shutil.copyfileobj(buffer, file)


def get_parquet_files_in_folder(folder: str | GCSPath) -> list[GCSPath]:
    """Lister opp parquetfiler i en "mappe" i en Google cloud bøtte.

    Nyttig hvis man har flere geoparquet filer, men som ikke har «hive» partisjonering.
    """
    file_paths = cast(Iterator[GCSPath], GCSPath(folder, protocol="gs").glob("**/*"))
    return list(filter(lambda p: p.suffix == ".parquet", file_paths))


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
        geo_series = cast(gpd.GeoSeries, gdf[geo_column])
        # Bruker geoarrow koding på alle kolonner hvor det er mulig, og WKB ellers.
        encoding = "geoarrow" if homogen_geometri(geo_series) else "WKB"

        if encoding == "geoarrow":
            field, geom_arr = construct_geometry_array(
                np.array(geo_series.array),
                field_name=geo_column,
                crs=geo_series.crs,
                interleaved=False,
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

        table = table.set_column(i, field, geom_arr)  # type: ignore[arg-type]

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    geo_metadata = _create_metadata(gdf, geometry_encoding_dict)
    metadata = table.schema.metadata if table.schema.metadata else {}
    metadata.update({b"geo": json.dumps(geo_metadata).encode("utf-8")})

    return table.replace_schema_metadata(metadata)


def _arrow_til_geopandas(
    arrow_table: pyarrow.Table,
    geometry_metadata: _GeoParquetMetadata,
    geometry_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Kopiert og tilpasset privat funksjon fra https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py."""
    geometry_columns = [
        col for col in geometry_metadata["columns"] if col in arrow_table.column_names
    ]
    result_column_names = list(arrow_table.slice(0, 0).to_pandas().columns)
    geometry_columns.sort(key=result_column_names.index)

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet file.  To read this file without geometry columns,
            use dapla.read_pandas() instead."""
        )

    geometry_column = (
        geometry_metadata["primary_column"] if not geometry_column else geometry_column
    )

    # Missing geometry likely indicates a subset of columns was read;
    if geometry_column not in geometry_columns:
        raise ValueError("Geometry column not in columns read from the Parquet file.")

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
                    extension_name="geoarrow." + column_metadata["encoding"],
                ),
                crs=crs,
            )

        df.insert(result_column_names.index(column), column, geom_arr)

    return gpd.GeoDataFrame(df, geometry=geometry_column)


def _create_metadata(
    gdf: gpd.GeoDataFrame, geometry_encoding: dict[str, str]
) -> _GeoParquetMetadata:
    schema_version = "1.1.0"

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
