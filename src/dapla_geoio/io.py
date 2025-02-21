from __future__ import annotations

import io
import json
import shutil
import sys
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyogrio
import pyogrio._err
import pyogrio.errors
import shapely
from dapla import AuthClient
from dapla import FileClient
from fsspec.implementations.arrow import ArrowFSWrapper
from geopandas.array import from_shapely
from geopandas.array import from_wkb
from geopandas.io._geoarrow import GEOARROW_ENCODINGS
from geopandas.io._geoarrow import construct_geometry_array
from geopandas.io._geoarrow import construct_shapely_array
from geopandas.io._geoarrow import construct_wkb_array
from pyarrow import fs
from pyarrow import parquet

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Required
    from typing import Self
    from typing import TypedDict
else:
    from strenum import StrEnum
    from typing_extensions import Required
    from typing_extensions import Self
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pyarrow._stubs_typing import FilterTuple


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


class BoundingBox(NamedTuple):
    """Avgrensningsboks etter Geojson rekkefølge."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def get_parquet_bbox_filter(
        self: Self, geo_metadata: _GeoParquetMetadata, geometry_column: str
    ) -> pc.Expression:
        """Lager Pyarrow filter for avgrensningsboks."""
        column_meta = geo_metadata["columns"][geometry_column]

        covering = column_meta.get("covering")
        if covering:
            bbox_column_name = covering["bbox"]["xmin"][0]
            return ~(
                (pc.field((bbox_column_name, "xmin")) > self.xmax)
                | (pc.field((bbox_column_name, "ymin")) > self.ymax)
                | (pc.field((bbox_column_name, "xmax")) < self.xmin)
                | (pc.field((bbox_column_name, "ymax")) < self.ymin)
            )

        elif column_meta["encoding"] == "point":
            return (
                (pc.field((geometry_column, "x")) >= self.xmin)
                & (pc.field((geometry_column, "x")) <= self.xmax)
                & (pc.field((geometry_column, "y")) >= self.ymin)
                & (pc.field((geometry_column, "y")) <= self.ymax)
            )

        else:
            raise ValueError(
                "Specifying 'bbox' not supported for this Parquet file (it should either "
                "have a bbox covering column or use 'point' encoding)."
            )

    def intersects(self: Self, other: Self) -> bool:
        """Returner sann hvis annen avgrensningsboks overlapper denne."""
        return not (
            (other.xmin > self.xmax)
            or (other.ymin > self.ymax)
            or (other.xmax < self.xmin)
            or (other.ymax < self.ymin)
        )

    def intersects_fragment(
        self: Self, fragment: ds.ParquetFileFragment, geometry_column: str
    ) -> bool:
        """Sjekker om et parquet-filfragment overlapper den gitte avgrensningsboksen."""
        fragment.ensure_complete_metadata()
        schema = fragment.physical_schema
        metadata = schema.metadata if schema.metadata else {}
        geo_metadata = _get_geometry_metadata(metadata)
        column_meta = geo_metadata["columns"][geometry_column]
        bbox = self.from_geo_metadata(column_meta)
        return self.intersects(bbox)

    @classmethod
    def from_geo_metadata(
        cls: type[Self], column_meta: _GeoParquetColumnMetadata
    ) -> Self:
        """Lager avgrensningsboks for angitte metadata."""
        try:
            bbox = column_meta["bbox"]
        except KeyError as err:
            raise ValueError("No bbox given in that dataset") from err

        return cls(*bbox)


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
    "parquet": FileFormat.PARQUET,
    "gpkg": FileFormat.GEOPACKAGE,
    "fgb": FileFormat.FLATGEOBUFFER,
    "json": FileFormat.GEOJSON,
    "geojson": FileFormat.GEOJSON,
    "shp": FileFormat.SHAPEFILE,
    "gdb": FileFormat.FILEGDB,
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
        return f"/vsigs/{gcs_path[len(prefix) :]}"

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

    return sorted([_GEOMETRY_TYPE_NAMES[idx] for idx in geometry_types])


def read_dataframe(
    path_or_paths: str | Iterable[str],
    file_format: FileFormat | None = None,
    columns: list[str] | None = None,
    bbox: Iterable[float] | BoundingBox | None = None,
    filters: list[FilterTuple] | list[list[FilterTuple]] | pc.Expression | None = None,
    geometry_column: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Leser inn en fil som innholder geometri til en Geopandas geodataframe.

    Støtter geoparquetfiler med WKB kodet geometri og partisjonerte geoparquetfiler.
    Bruker pyogrio til å lese andre filformater.
    """
    if file_format is None:
        if isinstance(path_or_paths, str):
            extension = path_or_paths.rpartition(".")[-1]
            file_format = _FILE_EXTENSTION2FORMAT.get(extension)

        elif all(path.endswith(".parquet") for path in path_or_paths):
            file_format = FileFormat.PARQUET

    if file_format == FileFormat.PARQUET:
        return _read_parquet(
            path_or_paths,
            columns=columns,
            bbox=bbox,
            filters=filters,
            geometry_column=geometry_column,
            **kwargs,
        )

    else:
        if not isinstance(path_or_paths, str):
            raise ValueError("Multiple paths are only supported for parquet format")

        set_gdal_auth()
        vsi_path = _ensure_gs_vsi_prefix(path_or_paths)

        try:
            return pyogrio.read_dataframe(
                vsi_path,
                bbox=bbox,
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

            path_or_paths = _remove_prefix(path_or_paths)
            filesystem = FileClient.get_gcs_file_system()
            with filesystem.open(path_or_paths, "rb") as buffer:
                return pyogrio.read_dataframe(
                    buffer,
                    bbox=bbox,
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
        file_format = _FILE_EXTENSTION2FORMAT.get(extension)

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
            raise ValueError("The Geojson-format supports only on geometry column.")

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


def _filter_fragments_with_bbox(
    fragments: Iterable[ds.ParquetFileFragment],
    bbox: BoundingBox,
    geometry_colum: str,
) -> list[ds.ParquetFileFragment]:
    return [
        fragment
        for fragment in fragments
        if bbox.intersects_fragment(fragment, geometry_colum)
    ]


def _read_parquet(
    path_or_paths: str | Iterable[str],
    *,
    columns: list[str] | None = None,
    bbox: BoundingBox | Iterable[float] | None = None,
    filters: list[FilterTuple] | list[list[FilterTuple]] | pc.Expression | None = None,
    geometry_column: str | None = None,
    schema: pyarrow.Schema | None = None,
) -> gpd.GeoDataFrame:
    if isinstance(path_or_paths, str):
        path_or_paths = _remove_prefix(path_or_paths)

    else:
        path_or_paths = [_remove_prefix(file) for file in path_or_paths]

    fileformat = ds.ParquetFileFormat()  # type: ignore[call-arg]
    partitioning = ds.HivePartitioning.discover(infer_dictionary=True)
    # pyarrow.dataset.dataset skal støtte fsspec filsystemer slik som gcsfs.GCSFileSystem,
    # ved å pakke inn fsspec filsystemer i fs.PyFileSystem(fs.FSSpecHandler(filesystem)),
    # men dette virker ikke for partisjonerte datasett.
    # Derfor bruker vi den Pyarrow spesifike implementasjonen.
    # https://github.com/apache/arrow/issues/30481
    arrow_filesystem = fs.GcsFileSystem()

    try:
        dataset: ds.FileSystemDataset = ds.dataset(
            path_or_paths,
            filesystem=arrow_filesystem,
            format=fileformat,
            schema=schema,
            partitioning=partitioning,
        )

    except pyarrow.ArrowTypeError as e:
        # Pyarrow er streng på at skjemaet i et partisjonert datasett skal være likt for alle filer,
        # når man ikke spesifiserer et.
        # I enkle tilfeller forsøker vi hente skjema fra den første filen vi finner, og bruker det.
        if schema is not None or not isinstance(path_or_paths, str):
            raise e
        filesystem = ArrowFSWrapper(arrow_filesystem)

        if not filesystem.isdir(path_or_paths):
            raise e

        fragment_paths = cast(
            list[str], filesystem.glob(f"{path_or_paths}/**/*.parquet", recursive=True)
        )
        parquet_file = parquet.ParquetFile(
            fragment_paths[0], filesystem=arrow_filesystem
        )

        schema = parquet_file.schema_arrow

        warnings.warn(
            "Pyarrow was unable to merge schema for partioned dataset,\n"
            "forced schema to be like first fragment found."
            "Orignal error:\n"
            f"{e}",
            stacklevel=4,
        )

        dataset = ds.dataset(
            path_or_paths,
            filesystem=arrow_filesystem,
            format=fileformat,
            schema=schema,
            partitioning=partitioning,
        )

    schema = dataset.schema
    metadata = schema.metadata if schema.metadata else {}

    if columns and b"pandas" in metadata:
        # RangeIndex can be represented as dict instead of column name
        index_columns = [
            col
            for col in _get_pandas_index_columns(metadata)
            if not isinstance(col, dict)
        ]
        columns = list(columns) + list(set(index_columns) - set(columns))

    geometry_metadata = _get_geometry_metadata(metadata)
    _validate_geometry_metadata(geometry_metadata, schema)

    primary_column = (
        geometry_metadata["primary_column"] if not geometry_column else geometry_column
    )

    if (
        columns and primary_column not in columns
    ) or primary_column not in geometry_metadata["columns"].keys():
        raise ValueError("Geometry column not in columns read from the Parquet file.")

    filters_expression = (
        parquet.filters_to_expression(filters) if filters is not None else None  # type: ignore[arg-type]
    )

    if bbox is not None:
        if not isinstance(bbox, BoundingBox):
            bbox = BoundingBox(*bbox)

        bbox_filter = bbox.get_parquet_bbox_filter(geometry_metadata, primary_column)

        filters_expression = (
            filters_expression & bbox_filter
            if filters_expression is not None
            else bbox_filter
        )

        fragments = _filter_fragments_with_bbox(
            dataset.get_fragments(), bbox=bbox, geometry_colum=primary_column
        )

        if not fragments:
            raise ValueError("No parts of the dataset overlaps the given bounding box")

        dataset = ds.FileSystemDataset(
            fragments,  # type: ignore[arg-type]
            schema=dataset.schema,
            format=dataset.format,
            filesystem=dataset.filesystem,
        )

    arrow_table = dataset.to_table(columns=columns, filter=filters_expression)

    if b"pandas" in metadata:
        new_metadata = arrow_table.schema.metadata or {}
        new_metadata.update({b"pandas": metadata[b"pandas"]})
        arrow_table = arrow_table.replace_schema_metadata(new_metadata)

    return _arrow_til_geopandas(arrow_table, geometry_metadata, geometry_column)


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


def _get_geometry_metadata(metadata: dict[bytes, bytes]) -> _GeoParquetMetadata:
    try:
        geo_metadata_bytes = metadata[b"geo"]

    except KeyError:
        raise ValueError(
            """No geometry metadata is present.
            To read a table without geometry, use dapla.read_pandas() instead"""
        ) from None

    return json.loads(geo_metadata_bytes.decode("utf-8"))  # type: ignore[no-any-return]


def _validate_geometry_metadata(
    geometadata: _GeoParquetMetadata, schema: pyarrow.Schema
) -> None:
    for col, column_metadata in geometadata["columns"].items():
        if col not in schema.names:
            raise ValueError("Geometry column in metadata don't exist in dataset")

        if column_metadata[
            "encoding"
        ] in GEOARROW_ENCODINGS and not pyarrow.types.is_struct(schema.field(col).type):
            warnings.warn(
                (
                    "Geoparquet files should not use the Geoarrow interleaved encoding.\n"
                    "A earlier version of this library used the wrong encoding.\n"
                    "The file will read, but you may be unable to filter the dataset."
                ),
                stacklevel=4,
            )

        if "covering" in column_metadata:
            covering = column_metadata["covering"]
            bbox = covering["bbox"]
            bbox_column_name = bbox["xmin"][0]
            if bbox_column_name not in schema.names:
                warnings.warn(
                    f"The geo metadata indicate that column '{col}' has a bounding box column named '{bbox_column_name}',"
                    "but this was not found in the dataset",
                    stacklevel=4,
                )
                del column_metadata["covering"]


def _get_pandas_index_columns(metadata: dict[bytes, bytes]) -> list[str | dict]:  # type: ignore[type-arg]
    return cast(
        list[str | dict],  # type: ignore[type-arg]
        json.loads(s=metadata[b"pandas"].decode("utf8"))["index_columns"],
    )


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
