from subprocess import run
from re import match
import sys

this_version = tuple(int(a) for a in sys.argv[1].split("."))

max_git_tag = max(
    [
        tuple(int(a) for a in line[1:].split("."))
        for line in run(["git", "tag"], capture_output=True)
        .stdout.decode("utf-8")
        .split("\n")
        if match("^v\d+\.\d+.\d+$", line)
    ]
)

if this_version > max_git_tag:
    sys.exit(0)
else:
    sys.exit(1)
