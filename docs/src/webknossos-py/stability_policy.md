# Stability Policy

This project adheres to [Semantic Versioning](http://semver.org/) regarding the Python API
compatibility, using the `MAJOR.MINOR.PATCH` version number format.
To allow fast development of the webKnossos server, the server-client interaction may need minor
version upgrades to ensure full functionality. Starting from v1 and above, minor and patch upgrades won't require
changes consuming the Python API as long as the following two points are followed:

1. To ensure compatibility across minor and patch upgrades, please import everything directly
   from the webknossos module, not from submodules, e.g. use
   ```python
   import webknossos as wk
   wk.Skeleton()
   # or
   from webknossos import Dataset
   Dataset()
   ```
2. Methods, functions, classes and variables prefixed with an underscore are not part of the public API
   and may change anytime.
