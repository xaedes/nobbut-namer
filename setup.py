
from setuptools import setup, find_packages
import glob, os

setup(
    name='nobbut_namer',
    version='0.0.1',
    author='xaedes',
    author_email='',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    entry_points={'console_scripts': ['nobbut_namer=nobbut_namer.main:main']},
)
