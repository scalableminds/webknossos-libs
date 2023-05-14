"""This module converts an image stack to a WEBKNOSSOS dataset."""

from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Tuple

import typer
from rich import print as rprint
from typing_extensions import Annotated

from webknossos import Dataset
from webknossos.utils import get_executor_for_args
from wkcuber.utils import DataFormat, DistributionStrategy

app = typer.Typer(invoke_without_command=True)


# *,
# map_filepath_to_layer_name: ConversionLayerMapping | ((Path) -> str) = ConversionLayerMapping.INSPECT_SINGLE_FILE,
# z_slices_sort_key: (Path) -> Any = natsort_keygen(),
# layer_category: LayerCategoryType | None = None,
# data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
# chunk_shape: Vec3IntLike | int | None = None,
# chunks_per_shard: Vec3IntLike | int | None = None,
# swap_xy: bool = False,
# flip_x: bool = False,
# flip_y: bool = False,
# flip_z: bool = False,
# use_bioformats: bool | None = None,
# max_layers: int = 20,
# batch_size: int | None = None


@app.callback()
def main(
    source: Annotated[Path, typer.Argument(help="Path to your image data.")],
    target: Annotated[
        Path, typer.Argument(help="Target path to save your WEBKNOSSOS dataset.")
    ],
    voxel_size: Annotated[
        Tuple[float, float, float],
        typer.Option(help="The size of one voxel in image data."),
    ],
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Dataformat to store the output dataset in.",
        ),
    ] = DataFormat.WKW,
    layer_name: Annotated[
        Optional[str], typer.Option(help="New name for the layer.")
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Compress the output dataset.")
    ] = False,
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
    job_ressources: Annotated[
        Optional[str],
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string (e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
):
    """Automatic detection of an image stack and conversion to a WEBKNOSSOS dataset."""

    rprint(f"Creating dataset [blue]{target}[/blue] from [blue]{source}[/blue] ...")

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_ressources=job_ressources,
    )

    with get_executor_for_args(args=executor_args) as executor:
        Dataset.from_images(
            source,
            target,
            voxel_size,
            name=layer_name,
            data_format=data_format,
            compress=compress,
            executor=executor,
        )

    rprint("[bold green]Done.[/bold green]")
