import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
  name = 'auglichem',
  packages=setuptools.find_packages(exclude=("docs",)), 
  version = '0.1.0',
  license = 'MIT',
  description = 'Data augmentation of molecules and crystals.',
  long_description = long_description,
  long_description_content_type = "text/markdown",
  author = 'Yuyang Wange, Rishikesh Magar, Cooper Lorsung, Hariharan Ramasubramanian, Chen Liang, Peiyuan Li, Amir Barati Farimani',
  author_email = 'clorsung@andrew.cmu.edu',
  url = 'https://github.com/BaratiLab/AugLiChem',
  classifiers = [
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    "Operating System :: OS Independent",
  ],
  python_requires = '>=3.8',
  install_requires = ['numpy>=1.21.1',
                      'pytest>=6.0.1',
                      'pytest-cov',
                      'sklearn',
                      'pandas>=1.3.1',
                      'matplotlib>=3.4.2',
                      'pyyaml>=5.4.1',
                      'tqdm',
                      'ase>=3.22.0',
                      'pymatgen>=2022.0.11',
                      'rdflib',

                      'rdkit-pypi',
                      'tensorboard>=2.4.1',
                     ],
)
