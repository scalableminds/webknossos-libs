"""This module takes care of downsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated

import typer
from upath import UPath

from ..dataset import SamplingModes
from ..geometry import Mag
from ..utils import get_executor_for_args
from ._utils import (
    DistributionStrategy,
    SamplingMode,
    open_dataset,
    parse_mag,
)


def main(
    *,
    target: Annotated[
        str,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset, or URL to a dataset on a WEBKNOSSOS server.",
            show_default=False,
        ),
    ],
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the layer to downsample (if not provided, all layers are downsampled)."
        ),
    ] = None,
    coarsest_mag: Annotated[
        Mag | None,
        typer.Option(
            help="Mag to stop downsampling at. \
Should be number or minus separated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
        ),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(
            help="Authentication token for WEBKNOSSOS instance "
            "(https://webknossos.org/auth/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
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
        str | None,
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Downsample your WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    sampling_mode_parsed = SamplingModes.parse(sampling_mode.value)

    with open_dataset(UPath(target), annotation_ok=False, token=token) as dataset:
        with get_executor_for_args(args=executor_args) as executor:
            if layer_name is None:
                dataset.downsample(
                    coarsest_mag=coarsest_mag,
                    sampling_mode=sampling_mode_parsed,
                    executor=executor,
                )
            else:
                dataset.get_layer(layer_name).downsample(
                    coarsest_mag=coarsest_mag,
                    sampling_mode=sampling_mode_parsed,
                    executor=executor,
                )
