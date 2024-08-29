# Annotation File to OME-TIFF

This example shows how to turn a volume annotation downloaded from WEBKNOSSOS into a OME-TIFF. When [manually downloading a WEBKNOSSOS annotation](/webknossos/export.html#data-export-through-the-ui) through the UI you end up with a ZIP file containing the volume segmentation in the WKW-format (and potentially any skeleton annotation).

As an alternative to manually downloading annotation files, have a look at streaming the data directly from the remote serve, e.g., in [this example](./download_segments.md).

```python
--8<--
webknossos/examples/load_annotation_from_file.py
--8<--
```
