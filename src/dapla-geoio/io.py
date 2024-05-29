import warnings
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pyogrio
import json
from pyarrow import parquet
from geopandas.array import from_shapely, from_wkb
from geopandas.io.geoarrow import construct_shapely_array
from geopandas.io.geoarrow import geopandas_to_arrow
from dapla import AuthClient, FileClient

filesystem = FileClient.get_gcs_file_system()

def set_gdal_auth() -> None:
    credentials = AuthClient.fetch_google_credentials()

    pyogrio.set_gdal_config_options({
        "GDAL_HTTP_HEADERS": f"Authorization: Bearer {credentials.token}",
        "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": True
    })


def _remove_gcs_uri_prefix(gcs_path: str) -> str:
    if gcs_path.startswith(prefix := "gs:/"):
        return gcs_path[len(prefix):]

    return gcs_path


def _ensure_gs_vsi_prefix(gcs_path: str) -> str:
    if "/vsigs/" in gcs_path:
        return gcs_path

    if gcs_path.startswith(prefix := "gs:/"):
        return f"/vsigs/{gcs_path[len(prefix):]}"

    return f"/vsigs/{gcs_path}"


def homogen_geometri(geoserie: gpd.GeoSeries) -> bool:
    """Sjekker at alle elementer i serien har lik geometritype og ikke er av typen GeometryCollection"""
    notnamaske = geoserie.notna()
    if not notnamaske.any():
        raise RuntimeError(f"Geometrikolonnen {geoserie.name} innholder kun tomme rader")

    notna = geoserie[notnamaske]
    første = notna.iat[0]
    return første.geom_type != "GeometryCollection" and (notna.geom_type == første.geom_type).all()


def read_geodataframe(
    gcs_path: str | Iterable[str],
    file_format: str | None = None,
    columns: Iterable[str] | None = None,
    # filters: list[tuple | list[tuple]] | pyarrow.compute.Expression | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:

    if file_format is None:
        if isinstance(gcs_path, str):
            if gcs_path.endswith(".parquet"):
                file_format = "parquet"
        else:
            if all(path.endswith(".parquet") for path in gcs_path):
                file_format = "parquet"

    if file_format == "parquet":
        if isinstance(gcs_path, str):
            gcs_path = FileClient._remove_gcs_uri_prefix(gcs_path)
        else:
            gcs_path = (FileClient._remove_gcs_uri_prefix(file) for file in gcs_path)

        arrow_table = parquet.ParquetDataset(
            gcs_path, filesystem=filesystem
        ).read(columns=columns, use_pandas_metadata=True)
        return _arrow_til_geopandas(arrow_table)

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

    if file_format is None and gcs_path.endswith(".parquet"):
        file_format = "parquet"

    if file_format == "parquet":
        table = _geopandas_to_arrow(gdf)
        with filesystem.open(gcs_path, mode="wb") as buffer:
            parquet.write_table(
                table,
                buffer,
                compression="snappy",
                coerce_timestamps="ms",
                **kwargs,
            )

    else:

        set_gdal_auth()
        path = _ensure_gs_vsi_prefix(gcs_path)

        pyogrio.write_dataframe(gdf, path, use_arrow=True, **kwargs)


def get_parquet_files_in_folder(folder):
    file_paths = FileClient.ls(folder)
    return list(filter(lambda p: p.endswith(".parquet"), file_paths))


def _geopandas_to_arrow(gdf):
    """
    Kopiert og tilpasset privat funkjson fra
    https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py.

    """
    from pyarrow import StructArray

    # create geo metadata before altering incoming data frame
    geo_metadata = _create_metadata(
        gdf
    )

    mask = gdf.dtypes == "geometry"
    alle_har_homogen_geometri = all(homogen_geometri(geo_column) for geo_column in gdf[mask])
    # Kun kolonner hvor alle rader har lik geometri type kan kodes med geoarrow.
    # Blandede geometrier og geometrisamlinger må kodes som wkb.
    # geopandas_to_arrow støtter ikke å velge geometry_encoding for hver enkelt kolonne.
    # Slik at hvis en kolonne må kodes som wkb, må alle kollonner kodes som wkb.

    geometry_encoding = "geoarrow" if alle_har_homogen_geometri else "wkb"

    table = geopandas_to_arrow(
        gdf, geometry_encoding=geometry_encoding
    )

    bounds = gdf.bounds
    bbox_array = StructArray().from_arrays(
        [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
        names=["xmin", "ymin", "xmax", "ymax"],
    )
    table = table.append_column("bbox", bbox_array)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": json.dumps(metadata).encode("utf-8")})

    return table.replace_schema_metadata(metadata)


def _arrow_til_geopandas(arrow_table):
    """
    Kopiert og tilpasset privat funkjson fra
    https://github.com/geopandas/geopandas/blob/9ad28395c0b094dbddd282a5bdf44900fe6650a1/geopandas/io/arrow.py.
    """
    df = arrow_table.to_pandas()

    geometry_metadata = json.loads(
        arrow_table.schema.metadata[b'geo'].decode("utf-8")
    )

    geometry_columns = df.columns.intersection(geometry_metadata["columns"])

    if not geometry_columns:
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry = geometry_metadata["primary_column"]

    # Missing geometry likely indicates a subset of columns was read;
    # promote the first available geometry to the primary geometry.
    if geometry not in geometry_columns:
        geometry = geometry_columns[0]

        # if there are multiple non-primary geometry columns, raise a warning
        if len(geometry_columns) > 1:
            warnings.warn(
                "Multiple non-primary geometry columns read from Parquet "
                "file. The first column read was promoted to the primary geometry.",
                stacklevel=3,
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
            df[column] = from_shapely(
                construct_shapely_array(
                    arrow_table[column].combine_chunks(), "geoarrow." + column_metadata["encoding"]
                ),
                crs=crs,
            )

    return gpd.GeoDataFrame(df, geometry=geometry)


def _create_metadata(df):
    """ """

    schema_version = "1.0.0"

    # Construct metadata for each geometry
    column_metadata = {}
    for col in df.columns[df.dtypes == "geometry"]:
        series = df[col]

        geometry_types = _get_geometry_types(series)
        geometry_types_name = "geometry_types"

        crs = series.crs.to_json_dict()

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
        "primary_column": df._geometry_column_name,
        "columns": column_metadata,
        "version": schema_version,
        "creator": {"library": "geopandas", "version": gpd.__version__},
    }
