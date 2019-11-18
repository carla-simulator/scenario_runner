import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='srunner',
     version='0.1',
     scripts=[],
     author="Carla Team",
     author_email="carla.simulatort@gmail.com",
     description="The package in order to use scenarios for CARLA",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/carla-simulator/scenario_runner",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )