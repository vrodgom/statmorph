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

Usage example
-------------

The following example loads a background-subtracted image and its
associated segmentation map, stellar mask (optional) and variance
(optional), then calculates the morphological parameters of all the
labeled sources in the segmentation map, and finally prints some
properties of the first labeled source.

.. code:: python

    import numpy as np
    from astropy.io import fits
    import statmorph

    hdulist_image = fits.open('image.fits.gz')
    hdulist_segmap = fits.open('segmap.fits.gz')
    hdulist_stmask = fits.open('stmask.fits.gz')
    hdulist_weights = fits.open('weights.fits.gz')

    image = hdulist_image['PRIMARY'].data
    segmap = hdulist_segmap['PRIMARY'].data
    mask = np.asarray(hdulist_stmask['PRIMARY'].data, dtype=np.bool8)
    variance = hdulist_weights['PRIMARY'].data

    source_morphology = statmorph.source_morphology(
        image, segmap, mask=mask, variance=variance, lazy_evaluation=True)

    hdulist_image.close()
    hdulist_segmap.close()
    hdulist_stmask.close()
    hdulist_weights.close()

    # Print some properties of the first source in the segmentation map
    morph = source_morphology[0]
    quantities = [
        'rpetro_circ',
        'rpetro_ellip',
        'r_20',
        'r_80',
        'gini',
        'm20',
        'sn_per_pixel',
        'flag',
        'asymmetry',
        'concentration',
        'smoothness',
        'multimode',
        'intensity',
        'deviation',
        'half_light_radius',
        'rmax',
        'outer_asymmetry',
        'shape_asymmetry',
    ]

    start_all = time.time()
    for quantity in quantities:
        start = time.time()
        value = morph[quantity]
        print('%25s: %10.6f    (Time: %9.6f s)' % (
              quantity, value, time.time() - start))
    print('\nTotal time: %.6f s.' % (time.time() - start_all))

For Pan-STARRS galaxy *J235958.6+281704* in the g-band, this returns:

::

              rpetro_circ:  55.184543    (Time:  0.047625 s)
             rpetro_ellip:  97.532795    (Time:  1.222480 s)
                     r_20:  11.667072    (Time:  0.014146 s)
                     r_80:  60.150937    (Time:  0.030301 s)
                     gini:   0.574485    (Time:  0.137494 s)
                      m20:  -1.955428    (Time:  0.027181 s)
             sn_per_pixel:   3.981467    (Time:  0.001153 s)
                     flag:   0.000000    (Time:  0.000004 s)
                asymmetry:   0.147330    (Time:  0.100315 s)
            concentration:   3.203761    (Time:  0.008073 s)
               smoothness:   0.079198    (Time:  0.009048 s)
                multimode:   0.027788    (Time: 12.024174 s)
                intensity:   0.018720    (Time:  0.125747 s)
                deviation:   0.018686    (Time:  0.003108 s)
        half_light_radius:  25.610659    (Time:  0.286685 s)
                     rmax: 119.067208    (Time:  0.000003 s)
          outer_asymmetry:   0.191133    (Time:  0.170721 s)
          shape_asymmetry:   0.198903    (Time:  0.004484 s)

::

    Total time: 14.168886 s.

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
