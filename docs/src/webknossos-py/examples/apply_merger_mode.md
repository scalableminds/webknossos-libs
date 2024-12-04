# Apply Merger Mode Annotation

This example applies a merger mode annotation to a segmentation.

It downloads a [sample dataset](https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view) with a [sample annotation](https://webknossos.org/annotations/Explorational/6204d2cd010000db0003db91).
The annotation has been created with "merger mode", where skeleton trees are used to merge segments.
Read more about the [merger mode in the WEBKNOSSOS documentation](/webknossos/proofreading/merger_mode.html).
The example then uses the [Dataset API](dataset_usage.md) and [fastremap module](https://github.com/seung-lab/fastremap) for reading, remapping and rewriting the `segmentation` layer.

_This example additionally needs the fastremap package._

```python
--8<--
webknossos/examples/apply_merger_mode.py
--8<--
```
