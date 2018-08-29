from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='statmorph',
    version='0.3.2',
    description='Non-parametric morphological diagnostics of galaxy images',
    long_description=long_description,
    url='https://github.com/vrodgom/statmorph',
    author='Vicente Rodriguez-Gomez',
    author_email='vrg@jhu.edu',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='astronomy galaxies galaxy-morphology non-parametric',
    packages=['statmorph', 'statmorph.tests'],
    include_package_data=True,
    install_requires=['scipy>=0.19',
                      'scikit-image>=0.14',
                      'astropy>=2.0',
                      'photutils>=0.5'],
)
