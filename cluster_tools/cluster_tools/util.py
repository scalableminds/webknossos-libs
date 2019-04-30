import subprocess
import random
import string
import os
import threading
import time
import logging

def local_filename(filename=""):
    return os.path.join(os.getenv("CFUT_DIR", ".cfut"), filename)

def random_string(length=32, chars=(string.ascii_letters + string.digits)):
    return ''.join(random.choice(chars) for i in range(length))

def call(command, stdin=None):
    """Invokes a shell command as a subprocess, optionally with some
    data sent to the standard input. Returns the standard output data,
    the standard error, and the return code.
    """
    if stdin is not None:
        stdin_flag = subprocess.PIPE
    else:
        stdin_flag = None
    proc = subprocess.Popen(command, shell=True, stdin=stdin_flag,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(stdin)
    return stdout, stderr, proc.returncode

class CommandError(Exception):
    """Raised when a shell command exits abnormally."""
    def __init__(self, command, code, stderr):
        self.command = command
        self.code = code
        self.stderr = stderr

    def __str__(self):
        return "%s exited with status %i: %s" % (repr(self.command),
                                                 self.code,
                                                 repr(self.stderr))

def chcall(command, stdin=None):
    """Like ``call`` but raises an exception when the return code is
    nonzero. Only returns the stdout and stderr data.
    """
    stdout, stderr, code = call(command, stdin)
    if code != 0:
        raise CommandError(command, code, stderr)
    return stdout, stderr


def warn_after(job, seconds):
    '''
    Use as decorator to warn when a function is taking longer than {seconds} seconds.
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            exceeded_timeout = [False]
            start_time = time.time()

            def warn_function():
              logging.warn("Function {} is taking suspiciously long (longer than {} seconds)".format(job, seconds))
              exceeded_timeout[0] = True

            timer = threading.Timer(seconds, warn_function)
            timer.start()

            try:
                result = fn(*args, **kwargs)
                if exceeded_timeout[0]:
                    end_time = time.time()
                    logging.warn("Function {} succeeded after all (took {} seconds)".format(job, int(end_time - start_time)))
            finally:
                timer.cancel()
            return result
        return inner
    return outer
