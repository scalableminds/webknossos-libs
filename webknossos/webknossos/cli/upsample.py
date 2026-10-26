"""This module takes care of upsampling WEBKNOSSOS datasets."""

from typing import Annotated

import typer
from upath import UPath

from ..dataset import Layer, RemoteDataset, RemoteLayer, SamplingModes, TransferMode
from ..geometry import Mag
from ._utils import (
    AccessModeOption,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    SamplingMode,
    SamplingModeOption,
    TokenOption,
    TransferModeOption,
    get_executor_for_args,
    get_readable_layer,
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
    sampling_mode: SamplingModeOption = SamplingMode.ANISOTROPIC,
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
    token: TokenOption = None,
    jobs: JobsOption = None,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
    transfer_mode: TransferModeOption = TransferMode.AUTO,
    access_mode: AccessModeOption = None,
) -> None:
    """Upsample your WEBKNOSSOS dataset."""

    mode = SamplingModes.parse(sampling_mode.value)

    with open_dataset(
        UPath(source), annotation_ok=False, token=token, access_mode=access_mode
    ) as dataset:
        is_remote = isinstance(dataset, RemoteDataset)
        extra_kwargs: dict = {"transfer_mode": transfer_mode} if is_remote else {}
        layers: list[Layer | RemoteLayer] = (
            list(dataset.layers.values())
            if layer_name is None
            else [dataset.get_layer(layer_name)]
        )
        for layer in layers:
            if is_remote and access_mode is None:
                assert isinstance(layer, RemoteLayer)
                layer = get_readable_layer(layer, from_mag)
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
