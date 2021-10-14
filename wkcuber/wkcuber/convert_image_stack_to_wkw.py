import logging
from argparse import Namespace, ArgumentParser
from typing import Sequence

from .cubing import (
    get_channel_and_sample_count_and_dtype,
    cubing,
    create_parser as create_cubing_parser,
)
from .mag import Mag
from .utils import (
    add_isotropic_flag,
    setup_logging,
    add_sampling_mode_flag,
    get_executor_args,
    is_wk_compatible_layer_format,
    get_channel_and_sample_iters_for_wk_compatibility,
)


def create_parser() -> ArgumentParser:
    parser = create_cubing_parser()

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2. In case of anisotropic downsampling, "
        "the process is considered done when max(current_mag) >= max(max_mag) where max takes the "
        "largest dimension of the mag tuple x, y, z. For example, a maximum mag value of 8 (or 8-8-8) "
        "will stop the downsampling as soon as a magnification is produced for which one dimension is "
        "equal or larger than 8. "
        "The default value is calculated depending on the dataset size. In the lowest Mag, the size will be "
        "smaller than 100vx per dimension",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress this data",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--prefer_channels",
        help="If the data format is not clear, merge the data into a single layer with multiple wkw channels.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prefer_layers",
        help="If the data format is not clear, create multiple layers with one wkw channel.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--force_non_webknossos_format",
        help="Specifies if the data should be converted, although webKnossos cannot read the result.",
        default=False,
        action="store_true",
    )

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)
    add_isotropic_flag(parser)
    add_sampling_mode_flag(parser)

    return parser


def _handle_hierarchical_data(args: Namespace, dtype: str, sample_count: int) -> None:
    if not is_wk_compatible_layer_format(sample_count, dtype):
        if args.force_non_webknossos_format:
            logging.warning(
                "Chosen wkw format is incompatible with webKnossos. Proceeding anyways..."
            )
        else:
            raise AssertionError(
                "Chosen wkw format would not be compatible with webKnossos. If you wish to convert it anyway, use --force_wkw_format."
            )


def main(args: Namespace) -> None:
    setup_logging(args)

    if args.isotropic is not None:
        raise DeprecationWarning(
            "The flag 'isotropic' is deprecated. Consider using '--sampling_mode isotropic' instead."
        )

    arg_dict = vars(args)
    channel_count, sample_count, dtype = get_channel_and_sample_count_and_dtype(
        args.source_path
    )
    if arg_dict.get("dtype") is not None:
        dtype = args.get("dtype")

    assert not (
        args.prefer_layers and args.prefer_channels
    ), "Invalid configuration. You cannot use prefer_channels and prefer_layers simultaneously."

    channel_iter: Sequence = [None]
    sample_iter: Sequence = [None]
    if (
        arg_dict.get("channel_index") is not None
        or arg_dict.get("sample_index") is not None
    ):
        # The user selected an explicit channel or sample
        channel_iter = [arg_dict.get("channel_index")]
        sample_iter = [arg_dict.get("sample_index")]
    elif not args.prefer_layers and not args.prefer_channels:
        # user did not specify how to deal with data formats => make wk compatible
        channel_iter, sample_iter = get_channel_and_sample_iters_for_wk_compatibility(
            channel_count, sample_count, dtype
        )
    elif args.prefer_channels:
        if channel_count > 1 and sample_count > 1:
            # no ambiguity exists, since channel axis and sample axis exist.
            # Use natural hierarchy, so each channel is a layer and the samples are wkw channels
            _handle_hierarchical_data(args, dtype, sample_count)
            channel_iter = range(channel_count)
            sample_iter = [None]
        else:
            # not both axes exist, so we use the disambiguation strategy prefer channels
            if not is_wk_compatible_layer_format(channel_count * sample_count, dtype):
                if args.force_non_webknossos_format:
                    logging.warning(
                        "Chosen wkw format is incompatible with webKnossos. Proceeding anyways..."
                    )
                else:
                    raise AssertionError(
                        "Chosen wkw format would not be compatible with webKnossos. If you wish to convert it anyway, use --force_wkw_format."
                    )
            channel_iter = [None]
            sample_iter = [None]
    elif args.prefer_layers:
        if channel_count > 1 and sample_count > 1:
            # no ambiguity exists, since channel axis and sample axis exist.
            # Use natural hierarchy, so each channel is a layer and the samples are wkw channels
            _handle_hierarchical_data(args, dtype, sample_count)
            channel_iter = range(channel_count)
            sample_iter = [None]
        else:
            # not both axes exist, so we use the disambiguation strategy prefer layers
            channel_iter = range(channel_count)
            sample_iter = range(sample_count)

    layer_count = 0
    layers = []
    for channel_index in channel_iter:
        for sample_index in sample_iter:
            layers.append(
                cubing(
                    args.source_path,
                    args.target_path,
                    f"{args.layer_name}_{layer_count}"
                    if len(channel_iter) * len(sample_iter) > 1
                    else args.layer_name,
                    arg_dict.get("batch_size"),
                    channel_index,
                    sample_index,
                    arg_dict.get("dtype"),
                    args.target_mag,
                    args.wkw_file_len,
                    args.interpolation_mode,
                    args.start_z,
                    args.skip_first_z_slices,
                    args.pad,
                    args.scale,
                    args,
                )
            )
            layer_count += 1

    for layer in layers:
        if not args.no_compress:
            layer.get_mag(args.target_mag).compress(args=args)

        layer.downsample(
            from_mag=args.target_mag,
            max_mag=None if args.max_mag is None else Mag(args.max_mag),
            compress=not args.no_compress,
            sampling_mode=args.sampling_mode,
            args=get_executor_args(args),
        )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
