statmorph
=========

Python code for calculating non-parametric morphological diagnostics of
galaxy images.

Brief description
-----------------

For a given (background-subtracted) image and a corresponding segmentation map
indicating the source(s) of interest, this code calculates the following
morphological statistics for each source:

- Gini-M20 statistics (Lotz et al. 2004)
- Concentration, Asymmetry and Smoothness (CAS) statistics (Conselice 2003;
  Lotz et al. 2004)
- Multimode, Intensity and Deviation (MID) statistics (Freeman et al. 2013;
  Peth et al. 2016)
- Outer asymmetry and shape asymmetry (Wen et al. 2014; Pawlik et al. 2016)
- Sersic index (Sersic 1968)
- Some properties associated to the above statistics (Petrosian radii,
  half-light radii, etc.)

Although the Sersic index is, by definition, the opposite of a non-parametric
morphological quantity, it is included anyway due to its popularity.

This Python implementation is largely based on IDL code originally
written by Jennifer Lotz, Peter Freeman and Mike Peth, as well as Python code
by Greg Snyder. The main scientific reference is
`Lotz et al. (2004) <http://adsabs.harvard.edu/abs/2004AJ....128..163L>`_,
but a more complete list can be found in the *Citing* section.

Documentation
-------------

The documentation can be found on
`ReadTheDocs <http://statmorph.readthedocs.io/en/latest/>`_.

Tutorial / How to use
---------------------

Please see the
`statmorph tutorial <http://nbviewer.jupyter.org/github/vrodgom/statmorph/blob/master/notebooks/tutorial.ipynb>`_.

Installation
------------

The easiest way to install this package is within the Anaconda environment:

.. code:: bash

    conda install -c conda-forge statmorph

If you do not have Anaconda installed yet, you should have a look at
`astroconda <https://astroconda.readthedocs.io>`_.

Alternatively, assuming that you already have recent versions of scipy,
scikit-image, astropy and photutils installed, statmorph can also be
installed via PyPI:

.. code:: bash

    pip install statmorph

Finally, if you prefer a manual installation, download the latest release
from the `GitHub repository <https://github.com/vrodgom/statmorph>`_,
extract the contents of the zipfile, and run:

.. code:: bash

    python setup.py install

Authors
-------

- Vicente Rodriguez-Gomez (vrg [at] jhu.edu)
- Jennifer Lotz
- Greg Snyder

Acknowledgments
---------------

- We thank Peter Freeman and Mike Peth for their IDL implementation of the
  MID statistics.

Citing
------

If you use this code for a scientific publication, please cite the following
article:

- Rodriguez-Gomez et al. (in prep.)

In addition, the Python package can also be cited using its Zenodo record:

.. image:: https://zenodo.org/badge/95412529.svg
   :target: https://zenodo.org/badge/latestdoi/95412529

Finally, below we provide some of the main references that introduce the
morphological parameters implemented in this code. The following list is
provided as a starting point and is not meant to be exhaustive. Please
see the references within each publication for more information.

- Gini--M20 statistics:

  - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163
  - Snyder G. F. et al., 2015, MNRAS, 454, 1886

- Concentration, asymmetry and clumpiness (CAS) statistics:

  - Bershady M. A., Jangren A., Conselice C. J., 2000, AJ, 119, 2645
  - Conselice C. J., 2003, ApJS, 147, 1
  - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

- Multimode, intensity and deviation (MID) statistics:

  - Freeman P. E., Izbicki R., Lee A. B., Newman J. A., Conselice C. J.,
    Koekemoer A. M., Lotz J. M., Mozena M., 2013, MNRAS, 434, 282
  - Peth M. A. et al., 2016, MNRAS, 458, 963

- Outer asymmetry and shape asymmetry:

  - Wen Z. Z., Zheng X. Z., Xia An F., 2014, ApJ, 787, 130
  - Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

- Sersic index:

  - Sersic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronomico
    de Cordoba, Cordoba
  - Any textbook about galaxies

Disclaimer
----------

This package is not meant to be the "official" implementation of any
of the morphological statistics described above. Please contact the
authors of the original publications for a "reference" implementation.

Licensing
---------

Licensed under a 3-Clause BSD License.
