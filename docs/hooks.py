"""MkDocs hooks.

Some pages are symlinked in from other repositories (e.g. the changelogs), so we
cannot add front matter to them directly. Instead, mark them here.
"""

from mkdocs.structure.pages import Page

# Pages (docs_dir-relative) that should not show up in the search index.
SEARCH_EXCLUDED_PAGES = {
    "webknossos/CHANGELOG.released.md",
    "webknossos-py/changelog.md",
}


def on_page_markdown(markdown: str, page: Page, **kwargs: object) -> str:
    if page.file.src_uri in SEARCH_EXCLUDED_PAGES:
        page.meta["search"] = {"exclude": True}
    return markdown
