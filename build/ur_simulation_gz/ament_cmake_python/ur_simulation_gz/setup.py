from setuptools import find_packages
from setuptools import setup

setup(
    name='ur_simulation_gz',
    version='2.2.0',
    packages=find_packages(
        include=('ur_simulation_gz', 'ur_simulation_gz.*')),
)
