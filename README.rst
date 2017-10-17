statmorph
=========

Python code for calculating non-parametric morphological parameters of
galaxy images.

Description
-----------

For a given image and a corresponding segmentation map indicating the
source(s) of interest (labeled by positive integer numbers), this code
calculates the following morphological parameters for each source:

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
morphological parameter, it is included for comparison purposes.

This Python implementation is largely based on IDL code originally
written by Jennifer Lotz, Peter Freeman and Mike Peth, as well as Python code by
Greg Snyder. The main scientific reference is
`Lotz et al. (2004) <http://adsabs.harvard.edu/abs/2004AJ....128..163L>`_,
but a more complete list can be found in the *Citing* section.

Input
-----

The main interface to use the code is the `source_morphology` function.
As input, it requires:

- *image* : The image (2D array) containing the source(s) of interest.
- *segmap* : A segmentation map (2D array) of the same size as the image with
  different sources labeled by different positive integer numbers. A value of
  zero is reserved for the background.

Optionally, the function can also accept:

- *mask* : A 2D array (of the same size as the image) indicating the pixels
  that should be masked (e.g., to remove contamination from foreground stars).
- *variance* : A 2D array (of the same size as the image) representing the
  local variance of the image. This is usually the inverse of the "weight" map
  produced by *SExtractor* and similar software. If the variance is not
  provided, statmorph will calculate it.
- *psf* : A 2D array (usually smaller than the image) with the point spread
  function (PSF). This is convolved with the Sersic model in every step of the
  profile fitting, and typically makes the code slower by a factor of 2-3.

In addition, almost all of the parameters used in the calculation of the
morphological diagnostics can be specified by the user as keyword
arguments, although it is recommended to leave the default values alone.
For a complete list of keyword arguments, see the docstring of the
`SourceMorphology` class.

Output
------

The output of the `source_morphology` function is a list of
`SourceMorphology` objects, one for each labeled source, in which the
different morphological measurements can be accessed as keys or attributes.

Apart from the morphological parameters, statmorph also produces three
different "bad measurement" flags (where values of 0 and 1 indicate good
and bad measurements, respectively):

1. *flag* : indicates a problem with the basic morphological measurements
   (e.g., a discontinuous Gini segmentation map).
2. *flag_segmap* : indicates when the 3 segmentation maps (Gini, MID,
   shape asymmetry) are very different from each other.
3. *flag_sersic* : indicates if there was a problem/warning during the
   Sersic profile fitting. 

In general, users should enforce *flag* == 0. The other two are optional.

In addition to the flags described above, the output should
not be trusted when any of the measured distance scales (Petrosian radii,
half-light radii, etc.) is smaller than the radius at half-maximum of the PSF,
or when the signal-to-noise per pixel (``sn_per_pixel``) is lower than 2.5
(`Lotz et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...636..592L>`_).

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

Alternatively, assuming that you already have scikit-image, astropy and
photutils installed, statmorph can also be installed via PyPI:

.. code:: bash

    pip install statmorph

Finally, for a manual installation, download the latest release from the
`GitHub repository <https://github.com/vrodgom/statmorph>`_,
extract the contents of the zipfile, and run:

.. code:: bash

    python setup.py install

Authors
-------
- Vicente Rodriguez-Gomez (vrg [at] jhu.edu)

Acknowledgments
---------------

- Based on IDL and Python code by Jennifer Lotz, Greg Snyder, Peter
  Freeman and Mike Peth.

Citing
------

If you use this code for a scientific publication, please cite the following
*Monthly Notices of the Royal Astronomical Society* article:

- Rodriguez-Gomez et al. (in prep.)

In addition, the Python package can also be cited using its Zenodo record:

.. image:: https://zenodo.org/badge/95412529.svg
   :target: https://zenodo.org/badge/latestdoi/95412529

Finally, below we provide some of the main references that describe the
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
Also see the `LICENSE`.

Licensing
---------

Licensed under a 3-Clause BSD License.
