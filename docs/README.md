# Documentation

Run `docs/generate.sh` to open a server rendering the documentation.
To update docstrings restart the server, manually written pages in `src` are auto-reloaded.

To get a live-reloading server for the docstrings, run `docs/generate.sh --api`. This opens pdoc, which looks differently than the final result, but the actual contents are the same.

To produce the html in `out`, run `docs/generate.sh --persist`.


## Further links

* https://www.mkdocs.org
* https://pdoc.dev
* https://squidfunk.github.io/mkdocs-material
* https://facelessuser.github.io/pymdown-extensions
* https://python-markdown.github.io/extensions/#officially-supported-extensions
