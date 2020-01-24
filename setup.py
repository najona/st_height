#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020.01.23
@author: alisa
"""


import setuptools
from setuptools import find_packages

print(find_packages("st_height"))
setuptools.setup(
    name="st_height",
    author="Nadja Jonas",
    author_email="s7najona@uni-bonn.de",
    description="This will be something someday",
    url="https://github.com/najona/st_height",
    packages=find_packages("."),
    package_dir={"":"."},
    classifiers=["Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Development Status :: 1 - Planning"]
    
)
