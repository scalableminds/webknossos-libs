"""This module takes care of downsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated

import typer
from upath import UPath

from ..dataset import RemoteDataset, SamplingModes, TransferMode
from ..dataset.remote_dataset import RemoteAccessMode
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
Should be number or hyphen-separated string (e.g. 2 or 2-2-2).",
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
    transfer_mode: Annotated[
        TransferMode | None,
        typer.Option(
            help="The transfer mode to use. Required for remote datasets. "
            "Options: 'copy', 'move+symlink', 'symlink', 'http'.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = None,
    access_mode: Annotated[
        RemoteAccessMode | None,
        typer.Option(
            help="How to access the remote dataset's data. "
            "Defaults to 'direct_path' when --transfer-mode is not 'http', otherwise 'zarr_streaming'.",
            rich_help_panel="WEBKNOSSOS context",
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

    if access_mode is None:
        if transfer_mode is not None and transfer_mode != TransferMode.HTTP:
            access_mode = RemoteAccessMode.DIRECT_PATH
        else:
            access_mode = RemoteAccessMode.PROXY_PATH

    with open_dataset(
        UPath(target), annotation_ok=False, token=token, access_mode=access_mode
    ) as dataset:
        with get_executor_for_args(args=executor_args) as executor:
            if isinstance(dataset, RemoteDataset):
                if transfer_mode is None:
                    raise typer.BadParameter(
                        "--transfer-mode is required for remote datasets.",
                        param_hint="--transfer-mode",
                    )
                extra_kwargs: dict = {"transfer_mode": transfer_mode}
            else:
                extra_kwargs = {}
            if layer_name is None:
                dataset.downsample(
                    coarsest_mag=coarsest_mag,
                    sampling_mode=sampling_mode_parsed,
                    executor=executor,
                    **extra_kwargs,
                )
            else:
                dataset.get_layer(layer_name).downsample(
                    coarsest_mag=coarsest_mag,
                    sampling_mode=sampling_mode_parsed,
                    executor=executor,
                    **extra_kwargs,
                )
