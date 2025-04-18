# Adapted from:
# Author - Kasun Herath <kasunh01 at gmail.com>
# Source - https://github.com/kasun/python-tail

import os
import sys
import time
from collections.abc import Callable
from typing import Any


class Tail:
    """Represents a tail command."""

    def __init__(
        self, tailed_file: str, callback: Callable[[str], Any] = sys.stdout.write
    ) -> None:
        """Initiate a Tail instance.
        Check for file validity, assigns callback function to standard out.

        Arguments:
            tailed_file - File to be followed."""

        self.tailed_file = tailed_file
        self.callback = callback
        self.is_cancelled = False

    def follow(self, seconds: int = 1) -> None:
        """Do a tail follow. If a callback function is registered it is called with every new line.
        Else printed to standard out.

        Arguments:
            seconds - Number of seconds to wait between each iteration; Defaults to 1.
        """

        self.check_file_validity(self.tailed_file)
        with open(self.tailed_file, encoding="utf-8", errors="replace") as file_:
            # Don't seek, since we want to print the entire file here.
            while True:
                line = file_.readline()
                if not line:
                    if self.is_cancelled:
                        # Only break here so that the rest of the file is consumed
                        # even when the job result is already available.
                        return
                    curr_position = file_.tell()
                    file_.seek(curr_position)
                    time.sleep(seconds)
                else:
                    self.callback(line)

    def cancel(self) -> None:
        self.is_cancelled = True

    def register_callback(self, func: Callable[[str], Any]) -> None:
        """Overrides default callback function to provided function."""
        self.callback = func

    def check_file_validity(self, file_: str) -> None:
        """Check whether the a given file exists, readable and is a file"""
        if not os.access(file_, os.F_OK):
            raise TailError(f"File '{file_}' does not exist")
        if not os.access(file_, os.R_OK):
            raise TailError(f"File '{file_}' not readable")
        if os.path.isdir(file_):
            raise TailError(f"File '{file_}' is a directory")


class TailError(Exception):
    pass
