statmorph
=========

Python code for calculating non-parametric morphological diagnostics
of galaxy images.

Description
-----------

The purpose of this code is to calculate some well-known (and some less
well-known) non-parametric morphological parameters for a set of
labeled galaxies in an image provided by the user.

This Python implementation is loosely based on IDL code originally
written by Jennifer Lotz, Peter Freeman and Mike Peth, as well as Python code by
Greg Snyder. The main reference is
`Lotz et al. (2004) <http://adsabs.harvard.edu/abs/2004AJ....128..163L>`_,
but a more complete list can be found in the *Citing* section.

As input, this code requires:

- The image containing the source(s) of interest.
- A corresponding segmentation map with different sources labeled
  by different positive integer numbers. A value of zero is reserved
  for the background.

Optionally, the code can also accept:

- A 2D array indicating the pixels that should be masked (e.g., to
  remove contamination from foreground stars).
- The local variance of the image. This is usually the inverse of the
  so-called "weight" map produced by *SExtractor* and similar software.

In addition, most parameters used in the calculation of the
morphological parameters can be specified by the user as keyword
arguments. For a list of keyword arguments, please see the docstring
of the `SourceMorphology` class.

Besides the morphological parameters, this code also produces a ``flag``
that attempts to indicate bad measurements. This flag is activated,
for example, when the calculated Gini segmentation map is discontinuous,
but also in other non-ideal circumstances.

For the sake of simplicity, this code does not ask for the size of
the point spread function (PSF), the pixel size in arcsec, or the
exposure time of the observations. However, note the output should
not be trusted when any of the measured scales (Petrosian radii,
``r_20``, ``rmax``) is smaller than the size of the PSF, or when the
signal-to-noise per pixel (``sn_per_pixel``) is lower than 2.5
(`Lotz et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...636..592L>`_).

Dependencies
------------

All of the following dependencies are included in the
`astroconda <https://astroconda.readthedocs.io>`_ environment:

- numpy
- scipy
- scikit-image
- astropy
- photutils

How to use
-------------

Please see the `statmorph tutorial <http://nbviewer.jupyter.org/github/vrodgom/statmorph/blob/master/notebooks/tutorial.ipynb>`_.

Authors
-------
- Vicente Rodriguez-Gomez (vrg [at] jhu.edu)

Acknowledgments
---------------

- Based on IDL and Python code by Jennifer Lotz, Greg Snyder, Peter
  Freeman and Mike Peth.

Citing
------

If you use this code for scientific publication, please cite
the package using its Zenodo record:

.. image:: https://zenodo.org/badge/95412529.svg
   :target: https://zenodo.org/badge/latestdoi/95412529

In addition, below we provide some of the main references that should
be cited when using each of the morphological parameters. This list is
provided as a starting point and is not meant to be exhaustive. Please
see the references within each publication for a more complete list.

- Gini--M20 statistics:

  - Abraham R. G., van den Bergh S., Nair P., 2003, ApJ, 588, 218
  - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163.

- Concentration, asymmetry and clumpiness (CAS) statistics:

  - Bershady M. A., Jangren A., Conselice C. J., 2000, AJ, 119, 2645
  - Conselice C. J., 2003, ApJS, 147, 1
  - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163.

- Multimode, intensity and deviation (MID) statistics:

  - Freeman P. E., Izbicki R., Lee A. B., Newman J. A., Conselice C. J.,
    Koekemoer A. M., Lotz J. M., Mozena M., 2013, MNRAS, 434, 282
  - Peth M. A. et al., 2016, MNRAS, 458, 963

- Outer asymmetry:

  - Wen Z. Z., Zheng X. Z., Xia An F., 2014, ApJ, 787, 130
  - Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

- Shape asymmetry:

  - Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

Disclaimer
----------

This package is not meant to be the "official" implementation of any
of the morphological statistics described above. Please contact the
authors of the original publications for a "reference" implementation.
Also see the `LICENSE`.

Licensing
---------

- Licensed under a 3-Clause BSD License.
