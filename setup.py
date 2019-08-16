from setuptools import setup, find_packages

setup(
    name="wkcuber",
    packages=find_packages(exclude=("tests",)),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=["scipy", "numpy", "pillow", "pyyaml", "wkw", "cluster_tools==1.36", "natsort"],
    description="A cubing tool for webKnossos",
    author="Norman Rzepka",
    author_email="norman.rzepka@scalableminds.com",
    url="https://scalableminds.com",
)
