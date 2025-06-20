from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Required
from typing import Self
from typing import TypedDict

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs
from geopandas.io._geoarrow import GEOARROW_ENCODINGS
from pyarrow import parquet
from upath.implementations.cloud import GCSPath

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


class GeoParquetDataset(parquet.ParquetDataset):
    """Tilpasset versjon av ParquetDataset, som kan ta en avgrensningsboks, i tillegg til filter."""

    def __init__(
        self: Self,
        paths: Sequence[GCSPath],
        schema: pyarrow.Schema | None = None,
        geometry_column: str | None = None,
        filters: list[FilterTuple | list[FilterTuple]] | ds.Expression | None = None,
        bbox: Iterable[float] | BoundingBox | None = None,
    ) -> None:
        """Enklere versjon som kan ta en avgrensningsboks, i tillegg til filter.

        Hvis vi kommer over et partisjonert datasett som innholder et inkonsistent skjema,
        forsøker vi å tvinge skjemaet til den første fila vi finner over på de andre.
        Kan foreløpig kun lese filer på GCS.
        """
        fileformat = ds.ParquetFileFormat()  # type: ignore  [call-arg]
        filesystem = pyarrow.fs.GcsFileSystem()

        if len(paths) == 1 and paths[0].is_file():
            single_file = paths[0].path
            fragment = fileformat.make_fragment(single_file, filesystem)
            self._dataset = ds.FileSystemDataset(
                [fragment],
                schema=schema or fragment.physical_schema,
                format=fileformat,
                filesystem=filesystem,
            )

        else:
            partitioning = ds.HivePartitioning.discover(infer_dictionary=True)

            if len(paths) == 1 and paths[0].is_dir():
                str_paths: list[str] = [
                    path.path for path in paths[0].glob("**/*.parquet")
                ]
                base_dir: str | None = paths[0].path

            else:
                str_paths = [path.path for path in paths]
                base_dir = None

            try:
                self._dataset = ds.dataset(
                    str_paths,
                    schema=schema,
                    format=fileformat,
                    filesystem=filesystem,
                    partitioning=partitioning,
                    partition_base_dir=base_dir,
                )

            except pyarrow.ArrowTypeError as e:
                # Pyarrow er streng på at skjemaet i et partisjonert datasett skal være likt for alle filer,
                # når man ikke spesifiserer et.
                # I enkle tilfeller forsøker vi hente skjema fra den første filen vi finner, og bruker det.
                if schema is not None:
                    raise e

                fragment_paths = list(paths[0].glob("**/*.parquet"))

                if len(fragment_paths) <= 1:
                    raise e

                parquet_file = fileformat.make_fragment(
                    fragment_paths[0].path, filesystem=filesystem
                )
                schema = parquet_file.physical_schema

                warnings.warn(
                    "Pyarrow was unable to merge schema for partioned dataset,\n"
                    "forced schema to be like first fragment found. "
                    "Original error:\n"
                    f"{e}",
                    stacklevel=4,
                )

                self._dataset = ds.dataset(
                    str_paths,
                    filesystem=filesystem,
                    format=fileformat,
                    schema=schema,
                    partitioning=partitioning,
                )

        metadata = self.schema.metadata or {}

        self.geometry_metadata = _get_geometry_metadata(metadata)
        self._validate_geometry_metadata()

        self.primary_geometry_column = (
            self.geometry_metadata["primary_column"]
            if not geometry_column
            else geometry_column
        )

        self._base_dir = None
        if filters is None or isinstance(filters, ds.Expression):
            self._filter_expression = filters
        else:
            self._filter_expression = parquet.filters_to_expression(filters)

        if bbox is not None:
            bbox = BoundingBox(*bbox)

            bbox_filter = bbox.get_parquet_bbox_filter(
                self.geometry_metadata, self.primary_geometry_column
            )

            self._filter_expression = (
                self._filter_expression & bbox_filter
                if self._filter_expression is not None
                else bbox_filter
            )

            fragments = self._filter_fragments_with_bbox(
                self._dataset.get_fragments(),
                bbox=bbox,
                geometry_colum=self.primary_geometry_column,
            )

            if not fragments:
                raise ValueError(
                    "No parts of the dataset overlaps the given bounding box"
                )

            self._dataset = ds.FileSystemDataset(
                fragments,  # type: ignore[arg-type]
                schema=self._dataset.schema,
                format=self._dataset.format,
                filesystem=self._dataset.filesystem,
            )

    def read(
        self: Self,
        columns: list[str] | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = True,
    ) -> pyarrow.Table:
        """Leser datasett over til payarrow.Table.

        Sørger for å kopiere over geometadata.
        """
        table = super().read(
            columns, use_threads=use_threads, use_pandas_metadata=use_pandas_metadata
        )

        # Gjenoppretter geo metadata
        new_metadata = table.schema.metadata or {}
        new_metadata[b"geo"] = json.dumps(
            self.geometry_metadata, ensure_ascii=False
        ).encode("utf-8")
        table = table.replace_schema_metadata(new_metadata)

        return table

    def _validate_geometry_metadata(self: Self) -> None:
        schema = self.schema
        for col, column_metadata in self.geometry_metadata["columns"].items():
            if col not in schema.names:
                raise ValueError("Geometry column in metadata don't exist in dataset")

            column_field = schema.field(col)
            nested_type = column_field.type
            while pyarrow.types.is_list(nested_type):
                nested_type = nested_type.value_type

            if (
                column_metadata["encoding"] in GEOARROW_ENCODINGS
            ) and not pyarrow.types.is_struct(nested_type):
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

    @staticmethod
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


def _get_geometry_metadata(metadata: dict[bytes, bytes]) -> _GeoParquetMetadata:
    try:
        geo_metadata_bytes = metadata[b"geo"]

    except KeyError:
        raise ValueError(
            """No geometry metadata is present.
            To read a table without geometry, use dapla.read_pandas() instead"""
        ) from None

    return json.loads(geo_metadata_bytes.decode("utf-8"))  # type: ignore[no-any-return]
