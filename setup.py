from distutils.core import setup
setup(
  name = 'penlm',
  packages = ['penlm'],
  version = '0.1', 
  license='MIT',
  description = 'Penalized Linear Models for Classification and Regression',
  author = 'Edoardo Belli',
  author_email = 'iedobelli@gmail.com',
  url = 'https://github.com/bellibot/penlm',
  download_url = 'https://github.com/bellibot/penlm/archive/refs/tags/v1.0.0.tar.gz',
  keywords = ['Classification', 'Regression', 'Linear', 'Penalty'],
  install_requires=[
          'numpy',
          'joblib',
          'Pyomo',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',   # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'Intended Audience :: Developers',               # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.8',
  ],
)
