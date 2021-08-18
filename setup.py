import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
  name = 'auglichem',
  packages=setuptools.find_packages(), 
  version = '0.1.0',
  license = 'MIT',
  description = 'Data augmentation of molecules and crystals.e',
  long_description = long_description,
  long_description_content_type = "text/markdown",
  author = '',
  author_email = '',
  url = 'https://github.com/BaratiLab/AugLiChem',
  classifiers = [
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    "Operating System :: OS Independent",
  ],
  python_requires = '>=3.0, < 3.9',
  install_requires = ['numpy==1.21.1',
                      'pytest==6.0.1',
                      'pytest-cov',
                      'sklearn==0.24.2',
                      'pandas==1.3.1',
                      'matplotlib==3.4.2',
                      'tensorboard==2.4.1',
                      'pyyaml==5.4.1',
                      'rdkit==2021.03.4',

                      'torch==1.8.0',
                      'torchvision==0.2.2',
                      'torch-geometric==1.7.2',
                      'pytorch_scatter==2.0.',
                     ],
)
