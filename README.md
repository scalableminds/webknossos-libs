# webKnossos-libs
<img align="right" src="https://static.webknossos.org/images/oxalis.svg" alt="webKnossos Logo" />

## [webKnossos Python API](webknossos)
Python API for working with [webKnossos](https://webknossos.org) datasets, annotations and for webKnossos server interaction.

Use this for:
- reading/writing/manipulating raw 2D/3D image data and volume annotations/segmentation in webKnossos wrap (*.wkw) format
- handling/manupulation of webKnossos datasets
- reading/writing/manipulating webKnossos skeleton annotations (*.nml)
- uploading/downloading annotations from your webKnossos instance
- exectuing any of the wkCuber operations from below from your code


## [webKnossos cuber (wkcuber) CLI](wkcuber)
CLI tool to creating and manipulating [webKnossos](https://webknossos.org) [WKW](https://github.com/scalableminds/webknossos-wrap) datasets. WKW is a container format for efficiently storing large-scale 3D images as found in microscopy data.

Use this for:
- converting Tiff-stacks and other data formats for volume image data to webKnossos-compatible *.wkw files from the CLI
- up/downsampling of *.wkw files to different magnifiction levels (image pyramid) from the CLI
- compressing your *.wkw files to save disc space