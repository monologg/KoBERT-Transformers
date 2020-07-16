from setuptools import setup, find_packages
from kobert_transformers import __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(name='kobert-transformers',
      version=__version__,
      url='https://github.com/monologg/KoBERT-Transformers',
      license='Apache License 2.0',
      author='Jangwon Park',
      author_email='adieujw@gmail.com',
      description='Transformers library for KoBERT, DistilKoBERT',
      packages=find_packages(exclude=["test"]),
      long_description=open('./README.md', 'r', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      python_requires='>=3',
      zip_safe=False,
      include_package_data=True,
      classifiers=(
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ),
      install_requires=require_packages,
      keywords="distilkobert kobert bert pytorch transformers lightweight"
      )
