from setuptools import setup

setup(
    name='wkcuber',
    packages=[
        'wkcuber'
    ],
    package_dir={'wkcuber': 'wkcuber'},
    version='0.0.3',
    install_requires=[
        'scipy',
        'numpy',
        'pillow',
        'pyyaml',
        'wkw'
    ],
    description='A cubing tool for webKnossos',
    author='Norman Rzepka',
    author_email='norman@scm.io',
    url='http://scm.io'
)
