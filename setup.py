import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for kobert-transformers.")


with open("requirements.txt") as f:
    require_packages = [line.strip() for line in f]


with open(os.path.join("kobert_transformers", "version.txt")) as f:
    version = f.read().strip()


setup(
    name="kobert-transformers",
    version=version,
    url="https://github.com/monologg/KoBERT-Transformers",
    license="Apache License 2.0",
    author="Jangwon Park",
    author_email="adieujw@gmail.com",
    description="Transformers library for KoBERT, DistilKoBERT",
    packages=find_packages(exclude=["tests"]),
    long_description=open("./README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    zip_safe=False,
    include_package_data=True,
    install_requires=require_packages,
    keywords="distilkobert kobert bert pytorch transformers lightweight",
)
