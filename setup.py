import os
from setuptools import setup, find_packages


root_pkg = 'aiccf'
packages = [root_pkg] + [root_pkg + '.' + sub_pkg for sub_pkg in find_packages(root_pkg)]

setup(
    name = root_pkg,
    version = "1.0",
    author = "Luke Campagnola",
    author_email = "lukec@alleninstitute.org",
    description = ("User interface tools for interacting with Allen Institute CCF atlas."),
    license = "Allen Institute Software License",
    keywords = "common coordinate framework ccf allen institute brain atlas",
    packages=packages,
    classifiers=[
    ],
)
