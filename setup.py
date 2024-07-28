from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'penlm',
  packages = ['penlm'],
  version = 'v1.1.0', 
  license='MIT',
  description = 'Penalized Linear Models for Classification and Regression',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Edoardo Belli',
  author_email = 'iedobelli@gmail.com',
  url = 'https://github.com/bellibot/penlm',
  download_url = 'https://github.com/bellibot/penlm/archive/refs/tags/v1.1.0.tar.gz',
  keywords = ['Classification', 'Regression', 'Linear', 'Penalty'],
  python_requires='>=3.5',
  install_requires=[
          'numpy',
          'joblib',
          'Pyomo',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',   # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
  ],
)
