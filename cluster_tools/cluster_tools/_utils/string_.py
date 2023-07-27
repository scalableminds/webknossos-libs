import os
import random
import string

# The module name includes a _-suffix to avoid name clashes with the standard library string module.


def local_filename(filename: str = "") -> str:
    return os.path.join(os.getenv("CFUT_DIR", ".cfut"), filename)


# Instantiate a dedicated generator to avoid being dependent on
# the global seed which some external code might have set.
random_generator = random.Random()


def random_string(
    length: int = 32, chars: str = (string.ascii_letters + string.digits)
) -> str:
    return "".join(random_generator.choice(chars) for i in range(length))


def with_preliminary_postfix(name: str) -> str:
    return f"{name}.preliminary"
