# Intro

The `webknossos` Python package provides an API for working with [webKnossos](https://webknossos.org) datasets,
annotations, and for webKnossos server interaction.
To get started, check out the [installation instructions](installation.md).


## Features

- Easy-to-use dataset API for reading/writing/editing raw 2D/3D image data
  and volume annotations/segmentation in webKnossos wrap (*.wkw) format
    - Add/remove layers
    - Update metadata (`datasource-properties.json`)
    - Up/downsample layers
    - Compress layers
    - Add/remove magnifications
    - Execute any of the `wkCuber` operations from your code
- Manipulation of webKnossos skeleton annotations (*.nml) as Python objects
    - Access to nodes, comments, trees, bounding boxes, metadata, etc.
    - Create new skeleton annotation from Graph structures or Python objects
- Interaction, connection & scripting with your webKnossos instance over the REST API
    - Up- & downloading annotations and datasets

## Source Code

The `webknossos` Python package is [open-source on GitHub][https://github.com/scalableminds/webknossos-libs]. Feel free to report bugs there or open pull requests with your features and fixes.

## License
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)
Copyright [scalable minds](https://scalableminds.com)
