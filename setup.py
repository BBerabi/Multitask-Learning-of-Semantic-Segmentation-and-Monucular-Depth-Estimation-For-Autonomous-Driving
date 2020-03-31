#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DLAD_ex2_multitask',
    version='1.0',
    description='Exercise 2: Multitask Learning in Autonomous Driving Environment',
    install_requires=requirements,
    packages=find_packages()
)
