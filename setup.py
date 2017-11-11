import os
from setuptools import setup, find_packages

packages=find_packages('.')

setup(
    name = "aiccf",
    version = "0.0.1",
    author = "Luke Campagnola",
    author_email = "lukec@alleninstitute.org",
    description = ("User interface tools for interacting with Allen Institute CCF atlas."),
    license = "",
    keywords = "common coordinate framework ccf allen institute brain atlas",
    packages=packages,
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)


