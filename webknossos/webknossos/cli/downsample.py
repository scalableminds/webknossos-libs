"""This module takes care of downsampling WEBKNOSSOS datasets."""

from typing import Annotated

import typer
from upath import UPath

from ..dataset import RemoteDataset, SamplingModes, TransferMode
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
    target: Annotated[
        str,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset, or URL to a dataset on a WEBKNOSSOS server.",
            show_default=False,
        ),
    ],
    sampling_mode: SamplingModeOption = SamplingMode.ANISOTROPIC,
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
    token: TokenOption = None,
    jobs: JobsOption = None,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
    transfer_mode: TransferModeOption = TransferMode.AUTO,
    access_mode: AccessModeOption = None,
) -> None:
    """Downsample your WEBKNOSSOS dataset."""

    sampling_mode_parsed = SamplingModes.parse(sampling_mode.value)

    with open_dataset(
        UPath(target), annotation_ok=False, token=token, access_mode=access_mode
    ) as dataset:
        with get_executor_for_args(
            jobs=jobs,
            distribution_strategy=distribution_strategy,
            job_resources=job_resources,
        ) as executor:
            if isinstance(dataset, RemoteDataset):
                layers = (
                    list(dataset.layers.values())
                    if layer_name is None
                    else [dataset.get_layer(layer_name)]
                )
                for layer in layers:
                    if access_mode is None and len(layer.mags) > 0:
                        # The mag that RemoteLayer.downsample reads from by default
                        layer = get_readable_layer(layer, max(layer.mags.keys()))
                    layer.downsample(
                        coarsest_mag=coarsest_mag,
                        sampling_mode=sampling_mode_parsed,
                        executor=executor,
                        transfer_mode=transfer_mode,
                    )
            elif layer_name is None:
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
