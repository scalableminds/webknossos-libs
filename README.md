# webKnossos cuber

[![CircleCI Status](https://circleci.com/gh/scalableminds/webknossos-cuber.svg?&style=shield)](https://circleci.com/gh/scalableminds/webknossos-cuber)

Cubing tool for webKnossos

```
# Convert image files to wkw cubes
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.cubing --layer_name color /data/source/color /data/target
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.cubing --layer_name segmentation /data/source/segmentation /data/target

# Create lower resolutions
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.downsampling --layer_name color /data/target
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.downsampling --layer_name segmentation /data/target

# Compress data in-place (mostly useful for segmentation)
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.compress --layer_name segmentation /data/target

# Compress data copy (mostly useful for segmentation)
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.compress --layer_name segmentation /data/target /data/target_compress

# Create metadata
docker run -v <host path>:/data webknossos_cuber --rm scalableminds/webknossos-cuber:wkw wkcuber.metadata --name great_dataset --scale 11.24,11.24,25 /data/target
```