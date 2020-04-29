#!/usr/bin/python3

import setuptools

<<<<<<< HEAD
from os import path, system, chdir
=======
from os import path, system
>>>>>>> 783b5253507121b1fefc04f5f933b16c79bd23b0
from setuptools.command.install import install 
from setuptools import setup, find_packages 
from sys import platform

class extra_install(install):
    """Extra operations required for install"""
 	
    def initialize_options(self):
        super(extra_install, self).initialize_options()

    def run(self):
        "Do extra setup step"
        if platform == "linux" or platform == "linux2":
            # when building inside docker we dont need to be sudo. 
            # otherwise, we must run it as sudoer
            system("apt-get update && apt-get install --no-install-recommends -y python3.6 python3-pip build-essential")
        super(extra_install, self).run()

with open(path.join(path.dirname(__file__),"README.md"), "r") as f:
    long_description = f.read()

with open(path.join(path.dirname(__file__),"requirements.txt"), "r") as f:
    required = f.read().splitlines()

setuptools.setup(
    # general info
    name="scenario_runner",  
    version='0.9.8',
    author="carla team",
    license='MIT',
    author_email="carla.simulator@gmail.com",
    description="ScenarioRunner for CARLA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carla-simulator/scenario_runner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # - main modules -
    py_modules=["manual_control", "no_rendering_mode", "scenario_runner"],
    # - packages -
<<<<<<< HEAD
    packages=setuptools.find_packages(),
=======
    packages=setuptools.find_packages("."),
>>>>>>> 783b5253507121b1fefc04f5f933b16c79bd23b0
    # - extra data
    include_package_data=True,
    # - requirements 
    install_requires=required,
    python_requires='>=3.5',
    cmdclass={'install': extra_install},
 )

