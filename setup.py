from setuptools import setup, find_packages

setup(
    name="wkcuber",
    packages=find_packages(exclude=("tests",)),
    version="0.2.10",
    install_requires=["scipy", "numpy", "pillow", "pyyaml", "wkw", "cluster_tools>=1.19"],
    description="A cubing tool for webKnossos",
    author="Norman Rzepka",
    author_email="norman.rzepka@scalableminds.com",
    url="https://scalableminds.com",
)
