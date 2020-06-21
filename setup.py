# setup.py

from setuptools import setup


PACKAGE = "excl-select"
NAME = "excl-select"
DESCRIPTION = "Exclusive Lasso with stability selection and random group allocation."
AUTHOR = "Yuxin Sun"
AUTHOR_EMAIL = "yuxin.sun.13@ucl.ac.uk"
URL = "https://github.com/YuxinSun/Stability-Selection"
VERSION = __import__(PACKAGE).__version__


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="MIT",
    url=URL,
    packages=[PACKAGE],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    classifiers=[
        "Development Status :: 1",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
