# Intro

Python API for working with [webKnossos](https://webknossos.org) datasets, annotations, and for webKnossos server interaction.


## Features

- easy-to-use dataset API for reading/writing/editing raw 2D/3D image data and volume annotations/segmentation in webKnossos wrap (*.wkw) format
    - add/remove layers
    - update metadata (`datasource-properties.json`) 
    - up/downsample layers
    - compress layers 
    - add/remove magnifications
    - execute any of the `wkCuber` operations from your code
- manipulation of webKnossos skeleton annotations (*.nml) as Python objects
    - access to nodes, comments, trees, bounding boxes, metadata, etc.
    - create new skeleton annotation from Graph structures or Python objects
- interaction, connection & scripting with your webKnossos instance over the REST API
    - up- & downloading annotations and datasets


## License
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)
Copyright [scalable minds](https://scalableminds.com)
