"""
Extension for markbox to make all images lightbox

copyright @2015 Alicia Schep <aschep@gmail.com>

"""
import markdown
from markdown.treeprocessors import Treeprocessor
from markdown import Extension
from markdown.util import etree
import re
from copy import copy


GROUP_RE = r'(^(@\{(?P<lightbox>.+)\})(?P<description>.*))'
HIDDEN_RE = r'(^(!)(?P<description>.*))'

class LightboxImagesTreeprocessor(Treeprocessor):
    """ Lightbox Images Treeprocessor """
    def __init__(self, md, group = True):
        Treeprocessor.__init__(self, md)
        self.group_re = re.compile(GROUP_RE)
        self.hidden_re = re.compile(HIDDEN_RE)
        self.group = group
    def run(self, root):
        parent_map = {c:p for p in root.iter() for c in p}
        i = 0
        images = root.getiterator("img")
        for image in images:
            desc = image.attrib["alt"]
            h = self.hidden_re.match(desc)
            if h:
                desc = h.group("description")
                hidden = True
            else:
                hidden = False
            m = self.group_re.match(desc)
            if m:
                lb = m.group("lightbox")
                desc = m.group("description")
            elif self.group:
                lb = "all_images"
            else:
                lb = "image" + str(i)
            image.set("alt", desc)
            parent = parent_map[image]
            ix = list(parent).index(image)
            new_node = etree.Element('a')
            new_node.set("href",image.attrib["src"])
            new_node.set("data-lightbox", lb)
            new_node.set("data-title",desc)
            new_node.tail = copy(image.tail)
            parent.insert(ix, new_node)
            parent.remove(image)
            image.tail = markdown.util.AtomicString("")
            if not hidden:
                new_node.append(image) 
            i += 1



class LightboxImagesExtension(Extension):
    """
    LightboxImagesExtension
    Extension class for markdown
    """
    def __init__(self, **kwargs):
        self.config = {'group' : [True,"group all images into same lightbox"] }
        super(LightboxImagesExtension, self).__init__(**kwargs)
    def extendMarkdown(self, md, md_globals):
        lightbox_images = LightboxImagesTreeprocessor(md, self.getConfig('group'))
        md.treeprocessors.add("lightbox", lightbox_images, "_end")
        md.registerExtension(self)

def makeExtension(**kwargs):
    return LightboxImagesExtension(**kwargs)
