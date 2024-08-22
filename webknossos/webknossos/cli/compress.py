"""This module takes care of compressing WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from webknossos.geometry.mag import Mag

from ..dataset import Dataset
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, parse_mag, parse_path


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
        Optional[str],
        typer.Option(
            help="Name of the layer to be compressed. If not provided, all layers will be compressed.",
        ),
    ] = None,
    mag: Annotated[
        Optional[List[Mag]],
        typer.Option(
            help="Mags that should be compressed. "
            "Should be number or minus separated string (e.g. 2 or 2-2-2). "
            "For multiple mags type: --mag 1 --mag 2",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        Optional[str],
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Compress a given WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    ds = Dataset.open(target)
    if layer_name is None:
        layers = list(ds.layers.values())
    else:
        layers = [ds.get_layer(layer_name)]

    with get_executor_for_args(args=executor_args) as executor:
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
