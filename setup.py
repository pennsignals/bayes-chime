"""Setup file for bayes_chime
"""
__version__ = "0.1.0"
__author__ = "Predictive Healthcare @ Penn Medicine"

from setuptools import setup, find_packages

with open("requirements.txt", "r") as INP:
    REQUIREMENTS = INP.read()

with open("README.md", "r") as INP:
    LONE_DESCRIPTION = INP.read()

setup(
    name="bayes_chime",
    version=__version__,
    author=__author__,
    author_email="",
    description="Bayesian fit to SEIR model."
    " An extension to Penn Medicine's CHIME tool.",
    long_description=LONE_DESCRIPTION,
    url="https://github.com/pennsignals/chime_sims",
    project_urls={
        "Bug Reports": "https://github.com/pennsignals/chime_sims/issues",
        "Source": "https://github.com/pennsignals/chime_sims",
    },
    packages=["bayes_chime"],
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    keywords=[],
)
