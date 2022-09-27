import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="online-label-smoothing",
    version="1.0",
    description="Online Label Smoothing LOSS",
    author="ankandrew",
    packages=find_packages(exclude=["test*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=False,
)
