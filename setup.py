from setuptools import setup

setup(
    name='webknossos_cuber',
    packages=[
        'webknossos_cuber'
    ],
    package_dir={'webknossos_cuber': 'webknossos_cuber'},
    version='0.0.1',
    install_requires=[
        'scipy',
        'numpy',
        'pillow',
        'pyyaml'
    ],
    description='A cubing tool for webKnossos',
    author='Norman Rzepka',
    author_email='norman@scm.io',
    url='http://scm.io'
)
