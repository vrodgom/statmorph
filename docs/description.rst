
Brief description
=================

The main interface between the user and the code is the `source_morphology`
function, which calculates the morphological parameters of a set of sources.
Below we briefly describe the input and output of this function.

A more detailed description of the input parameters and the measurements
performed by statmorph can be found in the API reference, as well as in
`Rodriguez-Gomez et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_.
We also refer the user to the
`tutorial <notebooks/tutorial.html>`_,
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
An exception might be the ``cutout_extent`` parameter, which specifies the
extent of the region (as a multiple of the size of the appropriate segment
in the segmentation map) used by statmorph to perform the morphological
measurements. Since the segmentation map is user-defined, in some cases it
makes sense to increase the value of ``cutout_extent`` from 2.5 (the default)
to 5 or 10, depending on the sensitivity of the original segmentation map.

For a complete list of keyword arguments, please see the
`API Reference <api.html>`_.

Output
------

The output of the `source_morphology` function is a list of
`SourceMorphology` objects, one for each labeled source, in which the
different morphological measurements can be accessed as keys or attributes.

Apart from the morphological parameters, statmorph also returns three
quality flags:

- ``flag`` : indicates the quality of the basic morphological measurements.
  It can take one of the following values:

  - 0 (good): there were no problems with the measurements.
  - 1 (suspect): the Gini segmap is discontinuous (e.g., due to a secondary
    source that was not properly labeled/masked) or the Gini and MID segmaps
    are very different from each other (as determined by the
    ``segmap_overlap_ratio`` keyword argument).
  - 2 (bad): there were problems with the measurements (e.g., the asymmetry
    minimizer tried to exit the image boundaries). However, most measurements
    are attempted anyway and a non-null value (i.e., not -99) might be
    returned for most measurements.
  - 3 (n/a): not currently used.
  - 4 (catastrophic): this value is returned when even the most basic
    measurements would be futile (e.g., a source with a negative total flux).
    This replaces the ``flag_catastrophic`` from earlier versions of statmorph.

- ``flag_sersic`` : indicates the quality of the Sersic fit. Just like
  ``flag``, it can take the following values: 0 (good), 1 (suspect), 2 (bad),
  and 4 (catastrophic).

- ``flag_doublesersic`` : indicates the quality of the double Sersic fit.
  It can take values of 0 (good), 1 (suspect), 2 (bad), and 4 (catastrophic).
  This flag is only generated when statmorph is called with the option
  ``include_doublesersic = True``.


In general, users should enforce ``flag <= 1``, while ``flag_sersic <= 1``
and ``flag_doublesersic <= 1`` should only be imposed when the user is
actually interested in the corresponding model fits (which, naturally, can
fail when the model is not a good description of the data).

In addition to the flags described above, the output should usually
not be trusted when the smallest of the measured distance scales (``r20``)
is smaller than the radius at half-maximum of the PSF,
or when the signal-to-noise per pixel (``sn_per_pixel``) is lower than 2.5
(`Lotz et al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ApJ...636..592L>`_).
