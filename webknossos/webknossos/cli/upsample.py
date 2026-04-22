"""This module takes care of upsampling WEBKNOSSOS datasets."""

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
    source: Annotated[
        str,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset, or URL to a dataset on a WEBKNOSSOS server.",
            show_default=False,
        ),
    ],
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    from_mag: Annotated[
        Mag,
        typer.Option(
            help="Mag to start upsampling from. \
Should be number or hyphen-separated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
        ),
    ],
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the layer that should be upsampled.", show_default=False
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
    """Upsample your WEBKNOSSOS dataset."""

    mode = SamplingModes.parse(sampling_mode.value)

    if access_mode is None:
        if transfer_mode is not None and transfer_mode != TransferMode.HTTP:
            access_mode = RemoteAccessMode.DIRECT_PATH
        else:
            access_mode = RemoteAccessMode.PROXY_PATH

    with open_dataset(
        UPath(source), annotation_ok=False, token=token, access_mode=access_mode
    ) as dataset:
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
            for layer in dataset.layers.values():
                with get_executor_for_args(
                    jobs=jobs,
                    distribution_strategy=distribution_strategy,
                    job_resources=job_resources,
                ) as executor:
                    layer.upsample(
                        from_mag=from_mag,
                        sampling_mode=mode,
                        executor=executor,
                        **extra_kwargs,
                    )
        else:
            with get_executor_for_args(
                jobs=jobs,
                distribution_strategy=distribution_strategy,
                job_resources=job_resources,
            ) as executor:
                dataset.get_layer(layer_name).upsample(
                    from_mag=from_mag,
                    sampling_mode=mode,
                    executor=executor,
                    **extra_kwargs,
                )
