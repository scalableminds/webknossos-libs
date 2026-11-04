# Documentation

## Development
The documentation build needs the webknossos repository: its `docs` are rendered as part of this site, and its frontend sources (`frontend/`) are used for the frontend scripting API reference.
Either clone it to `webknossos-libs/docs/wk-repo`, e.g. with `git clone --depth 1 git@github.com:scalableminds/webknossos.git webknossos-libs/docs/wk-repo`, or link an existing checkout there with `ln -s /path/to/webknossos wk-repo`.
`generate.sh` then links `src/webknossos` to `wk-repo/docs`:
```
webknossos-libs/docs$ ls -l wk-repo src/webknossos
wk-repo -> /path/to/webknossos
src/webknossos -> ../wk-repo/docs
```
Alternatively, link only the `docs` directory of an existing checkout with `ln -s /path/to/webknossos/docs src/webknossos`; `generate.sh` then links `wk-repo` to that checkout.
Do not clone the full repository to `src/webknossos`.
If webknossos is checked out next to webknossos-libs (`../../webknossos` from `docs`), `generate.sh` finds it automatically.

Run `./generate.sh` to open a live-reloading server rendering the documentation.
Besides `uv`, this requires Node.js/`npm`: The frontend scripting API reference (`src/frontend-api`) is generated from the TypeScript sources of the webknossos repository (`wk-repo/frontend`) by [mkdocstrings-typescript](https://mkdocstrings.github.io/typescript/), which calls [TypeDoc](https://typedoc.org) (see `typedoc.json` and `package.json`).

## Production

Run `./generate.sh --persist` to produce the production website/HTML in `out`. Use GitHub Actions for building and deploying the website.


## Further links

* https://www.mkdocs.org
* https://squidfunk.github.io/mkdocs-material
* https://facelessuser.github.io/pymdown-extensions
* https://python-markdown.github.io/extensions/#officially-supported-extensions
* https://mkdocstrings.github.io/
