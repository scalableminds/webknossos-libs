from typing import List

from markdown.core import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.postprocessors import Postprocessor


class PdocTocPreprocessor(Preprocessor):
    def run(self, lines: List[str]):
        if lines[0] == "pdoc-toc:":
            meta_lines = []
            while lines:
                line = lines.pop(0)
                if line == "---":
                    break
                meta_lines.append(line)
            import yaml
            meta = yaml.safe_load("\n".join(meta_lines))
            self.md.pdoc_toc_tokens = meta["pdoc-toc"] or []

        return lines


class PdocTocPostprocessor(Postprocessor):
    def run(self, text: str):
        self.md.toc_tokens += self.md.pdoc_toc_tokens
        return text


class PdocTocExtension(Extension):
    def extendMarkdown(self, md: Markdown):
        md.registerExtension(self)
        self.md = md
        self.reset()
        md.preprocessors.register(PdocTocPreprocessor(md), "pdoctocpre", 100)
        md.postprocessors.register(PdocTocPostprocessor(md), "pdoctocpost", 0)

    def reset(self):
        self.md.pdoc_toc_tokens = []


def makeExtension(**kwargs):
    return PdocTocExtension(**kwargs)
