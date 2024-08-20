"""This module takes care of downsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, Optional

import typer
from typing_extensions import Annotated

from ..dataset import Dataset, SamplingModes
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, SamplingMode, parse_path


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
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    layer_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the layer to downsample (if not provided, all layers are downsampled)."
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
    """Downsample your WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    dataset = Dataset.open(target)
    with get_executor_for_args(args=executor_args) as executor:
        if layer_name is None:
            dataset.downsample(
                sampling_mode=SamplingModes.parse(sampling_mode.value),
                executor=executor,
            )
        else:
            layer = dataset.get_layer(layer_name)
            layer.downsample(executor=executor)
