
Description
===========

The main interface between the user and the code is the `source_morphology`
function, which calculates the morphological parameters of a set of sources.
Here we briefly describe the input and output of this function.

Input
-----

The only two *required* input parameters are the following:

- ``image`` : The image (2D array) containing the source(s) of interest.
- ``segmap`` : A segmentation map (2D array) of the same size as the image with
  different sources labeled by different positive integer numbers. A value of
  zero is reserved for the background.

Optionally, the function can also accept:

- ``mask`` : A 2D array (of the same size as the image) indicating the pixels
  that should be masked (e.g., to remove contamination from foreground stars).
- ``variance`` : A 2D array (of the same size as the image) representing the
  local variance of the image. This is usually the inverse of the "weight" map
  produced by *SExtractor* and similar software. If the variance is not
  provided, statmorph will calculate it.
- ``psf`` : A 2D array (usually smaller than the image) with the point spread
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

1. ``flag`` : indicates a problem with the basic morphological measurements
   (e.g., a discontinuous Gini segmentation map).
2. ``flag_segmap`` : indicates when the 3 segmentation maps (Gini, MID,
   shape asymmetry) are very different from each other.
3. ``flag_sersic`` : indicates if there was a problem/warning during the
   Sersic profile fitting. 

In general, users should enforce ``flag == 0``. The other two are optional.

In addition to the flags described above, the output should
not be trusted when any of the measured distance scales (Petrosian radii,
half-light radii, etc.) is smaller than the radius at half-maximum of the PSF,
or when the signal-to-noise per pixel (``sn_per_pixel``) is lower than 2.5
(`Lotz et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...636..592L>`_).
