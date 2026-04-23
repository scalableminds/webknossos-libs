"""This module takes care of downsampling WEBKNOSSOS datasets."""

from typing import Annotated

import typer
from upath import UPath

from ..dataset import RemoteDataset, SamplingModes, TransferMode
from ..dataset.remote_dataset import RemoteAccessMode
from ..geometry import Mag
from ._utils import (
    AccessModeOption,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    SamplingMode,
    get_executor_for_args,
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
            "(https://webknossos.org/account/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    jobs: JobsOption = None,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
    transfer_mode: Annotated[
        TransferMode | None,
        typer.Option(
            help="The transfer mode to use. Required for remote datasets. "
            "Options: 'copy', 'move+symlink', 'symlink', 'http'.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = None,
    access_mode: AccessModeOption = None,
) -> None:
    """Downsample your WEBKNOSSOS dataset."""

    sampling_mode_parsed = SamplingModes.parse(sampling_mode.value)

    if access_mode is None:
        if transfer_mode is not None and transfer_mode != TransferMode.HTTP:
            access_mode = RemoteAccessMode.DIRECT_PATH
        else:
            access_mode = RemoteAccessMode.PROXY_PATH

    with open_dataset(
        UPath(target), annotation_ok=False, token=token, access_mode=access_mode
    ) as dataset:
        with get_executor_for_args(
            jobs=jobs,
            distribution_strategy=distribution_strategy,
            job_resources=job_resources,
        ) as executor:
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
