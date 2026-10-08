# Stability Policy

This project uses the `MAJOR.MINOR.PATCH` version number format and follows
[Intended Effort Versioning (EffVer)](https://effver.org/) for the Python API:
the version number tells you how much effort an upgrade is expected to take,
not whether something is technically incompatible. This means that, unlike with
[Semantic Versioning](http://semver.org/), minor releases may contain small breaking changes.

## Public API

Only the following is covered by this policy:

1. Everything that is imported directly from the webknossos module, not from submodules, e.g.
   ```python
   import webknossos as wk
   wk.Skeleton()
   # or
   from webknossos import Dataset
   Dataset()
   ```
2. Methods, functions, classes and variables prefixed with an underscore are not part of the public API
   and may change anytime.

## Versions

- **Major** releases may contain breaking changes to commonly used parts of the API.
  They are listed in the _Breaking Changes_ section of the [changelog](./changelog.md),
  together with upgrade instructions.
- **Minor** releases add features and deprecate APIs. They may also contain small breaking changes
  to rarely used parts of the API, or changes needed to keep up with the WEBKNOSSOS server.
  Such changes are listed in the _Changed_ section of the changelog and start with **Breaking:**.
- **Patch** releases only contain bug fixes.

Whether a breaking change is common enough to require a major release is decided case by case
when it is added to the changelog. Where possible, APIs are deprecated with a warning first,
listed in the _Deprecated_ section, and removed in a later release.
To be notified of upcoming removals, watch for `DeprecationWarning`s in your code.
