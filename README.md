# Dapla GeoIO

[![PyPI](https://img.shields.io/pypi/v/ssb-dapla-geoio.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-dapla-geoio.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-dapla-geoio)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-dapla-geoio)][license]

[![Documentation](https://github.com/statisticsnorway/dapla-geoio/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/dapla-geoio/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_dapla-geoio&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_dapla-geoio&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-dapla-geoio/
[documentation]: https://statisticsnorway.github.io/dapla-geoio
[tests]: https://github.com/statisticsnorway/dapla-geoio/actions?workflow=Tests

[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_dapla-geoio
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_dapla-geoio
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

## Funksjonalitet

_Dapla geoio_ leser og skriver filer med geometri til og fra en `geopandas.geodataframe` på SSBs dataplatform Dapla.
Pakka kan lese og skrive geoparquetfiler med WKB kodet geometri. Den kan også lese partisjonerte parquet-filer. Støtte for [Geoarrow] kodet geometri er planlagt.
_Dapla geoio_ bruker [Pyogrio] til å lese og skrive til andre filformater, og kan derfor også lese og skrive til de formatene som Pyogrio kan. Testet med Geopackage og Shape-filer.

Hvis du kun behøver lese og skrive funksjonalitet er _Dapla geoio_ et lettere alternativ til [ssb-sgis]

## Installasjon

Du kan installere _Dapla GeoIO_ via [pip] fra [PyPI]:

```console
pip install ssb-dapla-geoio
```

## Usage

Please see the [Reference Guide] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Dapla GeoIO_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/dapla-geoio/issues
[pip]: https://pip.pypa.io/
[pyogrio]: https://pypi.org/project/pyogrio/
[ssb-sgis]: https://pypi.org/project/ssb-sgis/
[geoarrow]: https://geoarrow.org

<!-- github-only -->

[license]: https://github.com/statisticsnorway/dapla-geoio/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/dapla-geoio/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/dapla-geoio/reference.html
