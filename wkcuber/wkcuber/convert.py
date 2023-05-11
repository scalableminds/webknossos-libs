"""This module converts an image stack to a WEBKNOSSOS dataset."""

from pathlib import Path
from typing import Optional, Tuple

import typer
from rich import print as rprint
from typing_extensions import Annotated
from webknossos import Dataset

app = typer.Typer(invoke_without_command=True)


#  -h, --help            show this help message and exit
#   --pad                 Automatically pad image files at the bottom and right borders. Use this, when the input images don't have a common size, but have their origin
#                         at (0, 0).
#   --voxel_size VOXEL_SIZE, --scale VOXEL_SIZE, -s VOXEL_SIZE
#                         Voxel size of the dataset in nm (e.g. 11.2,11.2,25). --scale is deprecated
#   --verbose             Verbose output
#   --data_format {wkw,zarr}
#                         Data format for outputs to be stored.
#   --chunk_shape CHUNK_SHAPE, --chunk_size CHUNK_SHAPE
#                         Number of voxels to be stored as a chunk in the output format (e.g. `32` or `32,32,32`).
#   --chunks_per_shard CHUNKS_PER_SHARD
#                         Number of chunks to be stored as a shard in the output format (e.g. `32` or `32,32,32`).
#   --wkw_file_len CHUNKS_PER_SHARD
#                         [DEPRECATED] Please use `--chunks_per_shard` instead.
#   --max_mag MAX_MAG, -m MAX_MAG
#                         Max resolution to be downsampled. Needs to be a power of 2. In case of anisotropic downsampling, the process is considered done when
#                         max(current_mag) >= max(max_mag) where max takes the largest dimension of the mag tuple x, y, z. For example, a maximum mag value of 8 (or
#                         8-8-8) will stop the downsampling as soon as a magnification is produced for which one dimension is equal or larger than 8. The default value is
#                         calculated depending on the dataset size. In the lowest Mag, the size will be smaller than 100vx per dimension
#   --no_compress         Don't compress this data
#   --version             show program's version number and exit
#   --name NAME, -n NAME  Name of the dataset
#   --isotropic           Activates isotropic downsampling. The default is anisotropic downsampling. Isotropic downsampling will always downsample each dimension with the
#                         factor 2.
#   --sampling_mode SAMPLING_MODE
#                         There are three different types: 'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel
#                         assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a
#                         basis for this method, the voxel_size from the datasource-properties.json is used. 'isotropic' - Each dimension is downsampled equally.
#                         'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.
#   --jobs JOBS, -j JOBS  Number of processes to be spawned.
#   --distribution_strategy {slurm,kubernetes,multiprocessing,debug_sequential}
#                         Strategy to distribute the task across CPUs or nodes.
#   --job_resources JOB_RESOURCES
#                         Necessary when using slurm as distribution strategy. Should be a JSON string (e.g., --job_resources='{"mem": "10M"}')

# input_path: str | PathLike[Unknown],
# output_path: str | PathLike[Unknown],
# voxel_size: Tuple[float, float, float],
# name: str | None = None,
# *,
# map_filepath_to_layer_name: ConversionLayerMapping | ((Path) -> str) = ConversionLayerMapping.INSPECT_SINGLE_FILE,
# z_slices_sort_key: (Path) -> Any = natsort_keygen(),
# layer_category: LayerCategoryType | None = None,
# data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
# chunk_shape: Vec3IntLike | int | None = None,
# chunks_per_shard: Vec3IntLike | int | None = None,
# compress: bool = False,
# swap_xy: bool = False,
# flip_x: bool = False,
# flip_y: bool = False,
# flip_z: bool = False,
# use_bioformats: bool | None = None,
# max_layers: int = 20,
# batch_size: int | None = None,
# executor: Executor | None = None


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
        str,
        typer.Option(
            help="Dataformat to store the output dataset in.",
        ),
    ] = "wkw",
    layer_name: Annotated[
        Optional[str], typer.Option(help="New name for the layer.")
    ] = None,
    compress: Annotated[bool, typer.Option(help="Compress the output dataset.")] = True,
):
    """Automatic detection of an image stack and conversion to a WEBKNOSSOS dataset."""

    rprint(f"Creating dataset [blue]{target}[/blue] from [blue]{source}[/blue] ...")

    Dataset.from_images(
        source,
        target,
        voxel_size,
        name=layer_name,
        data_format=data_format,
        compress=compress,
    )

    rprint("[bold green]Done.[/bold green]")
