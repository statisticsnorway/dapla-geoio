"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Dapla GeoIO."""


if __name__ == "__main__":
    main(prog_name="dapla-geoio")  # pragma: no cover
