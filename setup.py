from setuptools import find_packages, setup

setup(name='cale-detection', version='0.1.0',
      description='Collaborative adjustment based on lacunarity and entropy',
      packages=find_packages(include=['cale', 'cale.*', 'mmdet', 'mmdet.*']),
      python_requires='>=3.9')
