"""
# webKnossos Client Overview

You can interact with the webknossos server via the built-in client functions of this package.
The following table shows how objects can be down- & uploaded:

|              | [Dataset](dataset/dataset.html#Dataset)                       | [Annotation](annotation/annotation.html#Annotation)                |
|--------------|---------------------------------------------------------------|--------------------------------------------------------------------|
| **Download** | [`download_dataset(…)`](client/download_dataset.html)         | [`open_annotation(…)`](annotation/annotation.html#open_annotation) |
| **Upload**   | [`my_dataset.upload(…)`](dataset/dataset.html#Dataset.upload) |                                                                    |

Check [this page for information about authentication & server context specicification](client/context.html).

Additionally, one can [get user-specific information, see this page for more information](client/user.html).

"""


from .context import webknossos_context
from .download_dataset import download_dataset
from .user import User
