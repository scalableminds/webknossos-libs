import subprocess


def call(command: str, stdin: str | None = None) -> tuple[str, str, int]:
    """Invokes a shell command as a subprocess, optionally with some
    data sent to the standard input. Returns the standard output data,
    the standard error, and the return code.
    """
    if stdin is not None:
        stdin_flag = subprocess.PIPE
    else:
        stdin_flag = None
    p = subprocess.run(
        command,
        stdin=stdin_flag,
        check=False,
        shell=True,
        capture_output=True,
        text=True,
    )
    return p.stdout, p.stderr, p.returncode


class CommandError(Exception):
    """Raised when a shell command exits abnormally."""

    def __init__(self, command: str, code: int, stderr: str):
        self.command = command
        self.code = code
        self.stderr = stderr

    def __str__(self) -> str:
        return f"{self.command!r} exited with status {self.code}: {self.stderr!r}"


def chcall(command: str, stdin: str | None = None) -> tuple[str, str]:
    """Like ``call`` but raises an exception when the return code is
    nonzero. Only returns the stdout and stderr data.
    """
    stdout, stderr, code = call(command, stdin)
    if code != 0:
        raise CommandError(command, code, stderr)
    return stdout, stderr
