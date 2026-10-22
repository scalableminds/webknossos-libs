# Documentation

## Development
Either link or clone the webknossos repository to `webknossos-libs/docs/src/webknossos`, e.g. with `git clone --depth 1 git@github.com:scalableminds/webknossos.git webknossos-libs/docs/src/webknossos` or by creating a symlink with `ln -s ../path/to/webknossos/docs src/webknossos`
The symlink should look somewhat like this 
```
webknossos-libs/docs/src$ ls -l webknossos
webknossos -> ../../../webknossos/docs
```
Run `./generate.sh` to open a live-reloading server rendering the documentation.
Besides `uv`, this requires Node.js/`npm`: The frontend scripting API reference (`src/frontend-api`) is generated from the TypeScript sources of the webknossos repository (`wk-repo/frontend`) by [mkdocstrings-typescript](https://mkdocstrings.github.io/typescript/), which calls [TypeDoc](https://typedoc.org) (see `typedoc.json` and `package.json`).
`generate.sh` links `wk-repo` to the webknossos repository if only `src/webknossos` is linked.

## Production

Run `./generate.sh --persist` to produce the production website/HTML in `out`. Use GitHub Actions for building and deploying the website.


## Further links

* https://www.mkdocs.org
* https://squidfunk.github.io/mkdocs-material
* https://facelessuser.github.io/pymdown-extensions
* https://python-markdown.github.io/extensions/#officially-supported-extensions
* https://mkdocstrings.github.io/
