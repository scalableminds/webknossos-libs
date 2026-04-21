"""This module takes care of compressing WEBKNOSSOS datasets."""

from typing import Annotated, Any

import typer

from ..dataset import Dataset
from ..geometry.mag import Mag
from ._utils import (
    DEFAULT_JOBS,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    get_executor_for_args,
    parse_mag,
    parse_path,
)


def main(
    *,
    target: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the layer to be compressed. If not provided, all layers will be compressed.",
        ),
    ] = None,
    mag: Annotated[
        list[Mag] | None,
        typer.Option(
            help="Mags that should be compressed. "
            "Should be number or hyphen-separated string (e.g. 2 or 2-2-2). "
            "For multiple mags type: --mag 1 --mag 2",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = None,
    jobs: JobsOption = DEFAULT_JOBS,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
) -> None:
    """Compress a given WEBKNOSSOS dataset."""

    ds = Dataset.open(target)
    if layer_name is None:
        layers = list(ds.layers.values())
    else:
        layers = [ds.get_layer(layer_name)]

    with get_executor_for_args(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    ) as executor:
        for layer in layers:
            if mag is None:
                mags = list(layer.mags.values())
            else:
                mags = [layer.get_mag(mag) for mag in mag]
            for current_mag in mags:
                if not current_mag._is_compressed():
                    current_mag.compress(executor=executor)
                else:
                    typer.echo(f"Skipping {current_mag} as it is already compressed.")
