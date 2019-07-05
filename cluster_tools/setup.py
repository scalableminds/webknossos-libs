import os
from setuptools import setup, find_packages

def _read(fn):
    path = os.path.join(os.path.dirname(__file__), fn)
    return open(path).read()

setup(name='cluster_tools',
      version='v1.28',
      description='Utility library for easily distributing code execution on clusters',
      author='scalableminds',
      author_email='hello@scalableminds.com',
      url='https://github.com/scalableminds/cluster_tools',
      license='MIT',
      platforms='ALL',

      packages=find_packages(),
      install_requires=[
          'cloudpickle',
          'futures; python_version == "2.7"'
      ],

      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
      ],
)
