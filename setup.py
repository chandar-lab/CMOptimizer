import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

setup(name='cmoptimizer',
      version='1.0.0',
      description='Critical Momenta Optimizers',
      url='https://github.com/chandar-lab/CMOptimizer',
      author='Pranshu Malviya',
      author_email='pranshu.malviya@mila.quebec',
      license='MIT',
      install_requires=[
          'torch'
      ],
      packages=['cmoptimizer'],
      zip_safe=False)