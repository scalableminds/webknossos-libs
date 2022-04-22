from argparse import ArgumentParser
from os import environ

from webknossos import Dataset, webknossos_context
from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client._upload_dataset import DEFAULT_SIMULTANEOUS_UPLOADS

from ._internal.utils import add_verbose_flag, parse_path, setup_logging, setup_warnings


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Directory containing the source WKW dataset.",
        type=parse_path,
    )

    parser.add_argument(
        "--token",
        help="Auth token of the user on webKnossos",
        default=None,
    )

    parser.add_argument(
        "--url", help="Base url of the webKnossos instance", default=None
    )

    parser.add_argument(
        "--jobs",
        "-j",
        default=DEFAULT_SIMULTANEOUS_UPLOADS,
        type=int,
        help=f"Number of simultaneous upload processes. Defaults to {DEFAULT_SIMULTANEOUS_UPLOADS}.",
    )

    parser.add_argument(
        "--name",
        help="Specify a new name for the dataset. Defaults to the name specified in `datasource-properties.json`.",
        default=None,
    )

    add_verbose_flag(parser)

    return parser


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)
    url = (
        args.url
        if args.url is not None
        else environ.get("WK_URL", DEFAULT_WEBKNOSSOS_URL)
    )
    token = args.token if args.token is not None else environ.get("WK_TOKEN", None)
    assert (
        token is not None
    ), f"An auth token needs to be supplied either through the --token command line arg or the WK_TOKEN environment variable. Retrieve your auth token on {url}/auth/token."

    with webknossos_context(url=url, token=token):
        Dataset.open(args.source_path).upload(
            new_dataset_name=args.name, jobs=args.jobs
        )
