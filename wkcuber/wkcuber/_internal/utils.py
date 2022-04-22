from collections import namedtuple
from glob import iglob
from logging import getLogger
from math import ceil, floor
from os import environ
from typing import Generator, Sequence, Tuple

import numpy as np
import wkw
from webknossos import BoundingBox, DataFormat, Mag, Vec3Int
from webknossos.dataset.dataset import DEFAULT_DATA_FORMAT
from webknossos.dataset.defaults import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNKS_PER_SHARD
from webknossos.utils import *  # pylint: disable=unused-wildcard-import,wildcard-import

from .knossos import KnossosDataset

WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "mag", "header")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
FallbackArgs = namedtuple("FallbackArgs", ("distribution_strategy", "jobs"))

logger = getLogger(__name__)


Vec3 = Union[Tuple[int, int, int], np.ndarray]


def open_wkw(info: WkwDatasetInfo) -> wkw.Dataset:
    warn_deprecated("open_wkw", "Dataset.open")
    ds = wkw.Dataset.open(
        str(info.dataset_path / info.layer_name / str(info.mag)), info.header
    )
    return ds


def ensure_wkw(target_wkw_info: WkwDatasetInfo) -> None:
    warn_deprecated("ensure_wkw", "Dataset(..., exist_ok=True)")
    assert target_wkw_info.header is not None
    # Open will create the dataset if it doesn't exist yet
    target_wkw = open_wkw(target_wkw_info)
    target_wkw.close()


def parse_voxel_size(voxel_size: str) -> Tuple[float, ...]:
    try:
        return tuple(float(x) for x in voxel_size.split(","))
    except Exception as e:
        raise argparse.ArgumentTypeError("The voxel_size could not be parsed") from e


def parse_bounding_box(bbox_str: str) -> BoundingBox:
    try:
        return BoundingBox.from_csv(bbox_str)
    except Exception as e:
        raise argparse.ArgumentTypeError("The bounding box could not be parsed.") from e


def parse_padding(padding_str: str) -> Tuple[int, ...]:
    try:
        padding_tuple = tuple(int(x) for x in padding_str.split(","))
        assert len(padding_tuple) == 6, "Padding needs to have six components"
        return padding_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError("The padding could not be parsed") from e


def open_knossos(info: KnossosDatasetInfo) -> KnossosDataset:
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


class DeprecatedSizeAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if option_string == "--scale":
            warnings.warn(
                "[DEPRECATION] `--scale` is deprecated, please use `--voxel_size` instead.",
                DeprecationWarning,
            )
        setattr(namespace, self.dest, values)


def add_voxel_size_flag(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--voxel_size",
        "--scale",
        "-s",
        help="Voxel size of the dataset in nm (e.g. 11.2,11.2,25). --scale is deprecated",
        required=required,
        type=parse_voxel_size,
        action=DeprecatedSizeAction,
    )


def add_isotropic_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--isotropic",
        help="Activates isotropic downsampling. The default is anisotropic downsampling. "
        "Isotropic downsampling will always downsample each dimension with the factor 2.",
        dest="isotropic",
        default=None,
        action="store_true",
    )


def add_interpolation_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interpolation_mode",
        "-i",
        help="Interpolation mode (median, mode, nearest, bilinear or bicubic).",
        default="default",
    )


def add_sampling_mode_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampling_mode",
        help="There are three different types: "
        "'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the voxel_size from the datasource-properties.json is used. "
        "'isotropic' - Each dimension is downsampled equally. "
        "'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.",
        default="anisotropic",
    )


def is_wk_compatible_layer_format(channel_count: int, dtype: str) -> bool:
    return (channel_count == 1) or (channel_count == 3 and dtype == "uint8")


def get_channel_and_sample_iters_for_wk_compatibility(
    channel_count: int, sample_count: int, dtype: str
) -> Tuple[Sequence, Sequence]:
    if is_wk_compatible_layer_format(channel_count * sample_count, dtype):
        # combine all channel and samples into a single layer
        return ([None], [None])
    elif is_wk_compatible_layer_format(sample_count, dtype):
        # Convert each channel into a separate layer and convert each sample to wkw channels
        return (range(channel_count), [None])
    else:
        # Convert each channel and sample into a separate layer
        return (range(channel_count), range(sample_count))


def find_files(
    source_path: str, extensions: Iterable[str]
) -> Generator[Path, Any, None]:
    # Find all files with a matching file extension
    return (
        Path(f)
        for f in iglob(source_path, recursive=True)
        if any([f.lower().endswith(suffix) for suffix in extensions])
    )


# min_z and max_z are both inclusive
def get_regular_chunks(
    min_z: int, max_z: int, chunk_size: int
) -> Iterable[Iterable[int]]:
    i = floor(min_z / chunk_size) * chunk_size
    while i < ceil((max_z + 1) / chunk_size) * chunk_size:
        yield range(i, i + chunk_size)
        i += chunk_size


def add_distribution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jobs",
        "-j",
        default=cpu_count(),
        type=int,
        help="Number of processes to be spawned.",
    )

    parser.add_argument(
        "--distribution_strategy",
        default="multiprocessing",
        choices=["slurm", "kubernetes", "multiprocessing"],
        help="Strategy to distribute the task across CPUs or nodes.",
    )

    parser.add_argument(
        "--job_resources",
        default=None,
        help='Necessary when using slurm as distribution strategy. Should be a JSON string (e.g., --job_resources=\'{"mem": "10M"}\')',
    )


def add_batch_size_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of sections to buffer per job",
        type=int,
        default=DEFAULT_CHUNK_SIZE.z,
    )


def _parse_vec3_int(value: str) -> Vec3Int:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) == 1:
        return Vec3Int.full(parts[0])
    elif len(parts) == 3:
        return Vec3Int(*parts)
    else:
        raise TypeError(f"Cannot convert `{value}` to Vec3Int.")


def _parse_deprecated_file_len(value: str) -> Vec3Int:
    warn_deprecated("--wkw_file_len", "--chunks_per_shard")
    return _parse_vec3_int(value)


def _parse_data_format(value: str) -> DataFormat:
    return DataFormat(value)


def add_data_format_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_format",
        default=DEFAULT_DATA_FORMAT,
        type=_parse_data_format,
        choices=list(DataFormat),
        help="Data format for outputs to be stored.",
    )

    parser.add_argument(
        "--chunk_size",
        default=DEFAULT_CHUNK_SIZE,
        type=_parse_vec3_int,
        help="Number of voxels to be stored as a chunk in the output format (e.g. `32` or `32,32,32`).",
    )

    parser.add_argument(
        "--chunks_per_shard",
        default=DEFAULT_CHUNKS_PER_SHARD,
        type=_parse_vec3_int,
        help="Number of chunks to be stored as a shard in the output format (e.g. `32` or `32,32,32`).",
    )

    # Also stores
    parser.add_argument(
        "--wkw_file_len",
        default=None,
        type=_parse_deprecated_file_len,
        dest="chunks_per_shard",
        help="[DEPRECATED] Please use `--chunks_per_shard` instead.",
    )


def parse_path(value: str) -> Path:
    if (
        (value.startswith("http://") or value.startswith("https://"))
        and "HTTP_BASIC_USER" in environ
        and "HTTP_BASIC_PASSWORD" in environ
    ):
        import aiohttp

        return UPath(
            value,
            client_kwargs={
                "auth": aiohttp.BasicAuth(
                    environ["HTTP_BASIC_USER"], environ["HTTP_BASIC_PASSWORD"]
                )
            },
        )
    elif value.startswith("s3://") and "S3_ENDPOINT_URL" in environ:
        return UPath(
            value,
            client_kwargs={"endpoint_url": environ["S3_ENDPOINT_URL"]},
        )

    return UPath(value)


def pad_or_crop_to_size_and_topleft(
    cube_data: np.ndarray, target_size: np.ndarray, target_topleft: np.ndarray
) -> np.ndarray:
    """
    Given an numpy array and a target_size/target_topleft, the array
    will be padded so that it is within the bounding box descriped by topleft and size.
    If the input data is too large, the data will be cropped (evenly from opposite sides
    with the assumption that the most important data is in the center).
    """

    # Pad to size
    half_padding = (target_size - cube_data.shape) / 2
    half_padding = np.clip(half_padding, 0, None)
    left_padding = np.floor(half_padding).astype(np.uint32)
    right_padding = np.floor(half_padding).astype(np.uint32)

    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (left_padding[1], right_padding[1]),
            (left_padding[2], right_padding[2]),
            (0, 0),
        ),
    )

    # Potentially crop to size
    half_overflow = (cube_data.shape - target_size) / 2
    half_overflow = np.clip(half_overflow, 0, None)
    left_overflow = np.floor(half_overflow).astype(np.uint32)
    right_overflow = np.floor(half_overflow).astype(np.uint32)
    cube_data = cube_data[
        :,
        left_overflow[1] : cube_data.shape[1] - right_overflow[1],
        left_overflow[2] : cube_data.shape[2] - right_overflow[2],
        :,
    ]

    # Pad to topleft
    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (target_topleft[1], max(0, target_size[1] - cube_data.shape[1])),
            (target_topleft[2], max(0, target_size[2] - cube_data.shape[2])),
            (target_topleft[3], max(0, target_size[3] - cube_data.shape[3])),
        ),
    )

    return cube_data


def convert_mag1_offset(
    mag1_offset: Union[List, np.ndarray], target_mag: Mag
) -> np.ndarray:
    return np.array(mag1_offset) // target_mag.to_np()  # floor div


def get_executor_args(global_args: argparse.Namespace) -> argparse.Namespace:
    executor_args = argparse.Namespace()
    executor_args.jobs = global_args.jobs
    executor_args.distribution_strategy = global_args.distribution_strategy
    executor_args.job_resources = global_args.job_resources
    return executor_args
