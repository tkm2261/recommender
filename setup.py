#!/usr/bin/env python
# -*- coding:utf-8
from setuptools import setup, find_packages

import sys
sys.path.append('./recommend')

setup(
    name = "recommend",
    version = "0.1",
    packages = find_packages(),
    test_suite = 'test'
)