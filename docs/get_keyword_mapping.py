import inspect

import webknossos


print(
    "|".join(
        key + ":" + value.__module__
        for key, value in webknossos.__dict__.items()
        if getattr(value, "__module__", "").startswith("webknossos")
    )
)
