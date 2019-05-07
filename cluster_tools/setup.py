import os
from setuptools import setup

def _read(fn):
    path = os.path.join(os.path.dirname(__file__), fn)
    return open(path).read()

setup(name='cluster_tools',
      version='1.16',
      description='Utility library for easily distributing code execution on clusters',
      author='scalableminds',
      author_email='hello@scalableminds.com',
      url='https://github.com/scalableminds/cluster_tools',
      license='MIT',
      platforms='ALL',
      long_description=_read('README.md'),

      packages=['cluster_tools'],
      install_requires=[
          'cloudpickle',
          'futures',
      ],

      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
      ],
)
