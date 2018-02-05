# webKnossos cuber

[![CircleCI Status](https://circleci.com/gh/scalableminds/webknossos-cuber.svg?&style=shield)](https://circleci.com/gh/scalableminds/webknossos-cuber)

Cubing tool for webKnossos

Based on [knossos_cuber](https://github.com/knossos-project/knossos_cuber).

```
docker run -v <host path>:<docker path> webknossos_cuber -n <name> <source> <target>
```

Downsample only:
```
docker run -v <host path>:<docker path> webknossos_cuber -n <name> --downsample <source> <target>
```
