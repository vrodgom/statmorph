
Brief description
=================

The main interface between the user and the code is the `source_morphology`
function, which calculates the morphological parameters of a set of sources.
Below we briefly describe the input and output of this function.

A more detailed description of the input parameters and the measurements
performed by statmorph can be found in the API reference, as well as in
`Rodriguez-Gomez et al. (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_.
We also refer the user to the
`tutorial <http://nbviewer.jupyter.org/github/vrodgom/statmorph/blob/master/notebooks/tutorial.ipynb>`_,
which contains a more concrete (albeit simplified) usage example.

Input
-----

The main two *required* input parameters are the following:

- ``image`` : A *background-subtracted* image (2D array) containing the
  source(s) of interest.
- ``segmap`` : A segmentation map (2D array) of the same size as the image with
  different sources labeled by different positive integer numbers. A value of
  zero is reserved for the background.

In addition, *one* of the following two parameters is also required:

- ``weightmap`` : A 2D array (of the same size as the image) representing one
  standard deviation of each pixel value. This is also known as the "sigma"
  image and is related to the Poisson noise. If the weight map is not
  provided by the user, then it is computed internally using the ``gain``
  keyword argument.
- ``gain`` : A scalar that, when multiplied by the image, converts the image
  units into electrons/pixel. This parameter is required when ``weightmap``
  is not provided by the user.

Optionally, the function can also accept:

- ``mask`` : A 2D array (of the same size as the image) indicating the pixels
  that should be masked (e.g., to remove contamination from foreground stars).
- ``psf`` : A 2D array (usually smaller than the image) representing the point
  spread function (PSF). This is used when fitting Sersic profiles.

In addition, almost all of the parameters used in the calculation of the
morphological diagnostics can be specified by the user as keyword
arguments, although it is recommended to leave the default values alone.
For a complete list of keyword arguments, please see the
`API Reference <api.html>`_.

Output
------

The output of the `source_morphology` function is a list of
`SourceMorphology` objects, one for each labeled source, in which the
different morphological measurements can be accessed as keys or attributes.

Apart from the morphological parameters, statmorph also produces two
different "bad measurement" flags (where values of 0 and 1 indicate good
and bad measurements, respectively):

1. ``flag`` : indicates a problem with the basic morphological measurements
   (e.g., a discontinuous Gini segmentation map).
2. ``flag_sersic`` : indicates if there was a problem during the
   Sersic profile fitting. 

In general, users should enforce ``flag == 0``, while ``flag_sersic == 0``
should be applied only when actually interested in Sersic fits (which can
fail for merging galaxies and other "irregular" objects).

In addition to the flags described above, the output should
not be trusted when any of the measured distance scales (Petrosian radii,
half-light radii, etc.) is smaller than the radius at half-maximum of the PSF,
or when the signal-to-noise per pixel (``sn_per_pixel``) is lower than 2.5
(`Lotz et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...636..592L>`_).
