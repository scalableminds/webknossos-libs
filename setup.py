from setuptools import setup, find_packages

setup(
    name='wkcuber',
    packages=find_packages(exclude=("tests",)),
    version='0.1.5',
    install_requires=[
        'scipy',
        'numpy',
        'pillow',
        'pyyaml',
        'wkw'
    ],
    description='A cubing tool for webKnossos',
    author='Norman Rzepka',
    author_email='norman.rzepka@scalableminds.com',
    url='https://scalableminds.com'
)
