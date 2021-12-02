import os

from setuptools import find_packages, setup


def _read(fn):
    path = os.path.join(os.path.dirname(__file__), fn)
    return open(path).read()


setup(
    name="cluster_tools",
    version="0.0.0",  # filled by dunamai
    setup_requires=["setuptools_scm"],
    description="Utility library for easily distributing code execution on clusters",
    author="scalableminds",
    author_email="hello@scalableminds.com",
    url="https://github.com/scalableminds/cluster_tools",
    license="MIT",
    platforms="ALL",
    packages=find_packages(),
    install_requires=["cloudpickle", 'futures; python_version == "2.7"'],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
)
