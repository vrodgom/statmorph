# statmorph

Python code for calculating non-parametric morphological diagnostics
of galaxy images, based on definitions from
[Lotz et al. (2004)](http://adsabs.harvard.edu/abs/2004AJ....128..163L)
and other references listed below.

### Brief description ###

T.B.D.

### Dependencies ###

All of the following dependencies are included in the
[astroconda](https://astroconda.readthedocs.io) environment:

* Numpy
* Scipy
* Sklearn
* Astropy
* photutils

### Usage example ###

The following example loads an image and a segmentation map contained
in FITS files 'image.fits.gz' and 'segmap.fits.gz', then calculates the
morphologies of all the labeled sources in the segmentation map, and
finally prints the morphological parameters of the first source.

```python

import numpy as np
import statmorph

hdulist_image = fits.open('image.fits.gz')
hdulist_segmap = fits.open('segmap.fits.gz')

image = hdulist_image['PRIMARY'].data
segmap = hdulist_segmap['PRIMARY'].data

source_morphology = statmorph.source_morphology(image, segmap)

# Print properties of first source in the segmentation map
morph = source_morphology[0]
print('Gini:', morph.gini)
print('M20:', morph.m20)
print('Asymmetry:', morph.asymmetry)
print('Concentration:', morph.concentration)
print('Smoothness:', morph.smoothness)

hdulist_image.close()
hdulist_segmap.close()

```

For Pan-STARRS galaxy **J235958.6+281704** in the g-band, this returns:

```
Gini: 0.582655879166
M20: -1.94720368966
Asymmetry: 0.17009201935
Concentration: 3.20757730314
Smoothness: 0.0979850445916

```

### Authors ###
* Vicente Rodriguez-Gomez (vrg [at] jhu.edu)

### Contributors ###
* T.B.D.

### Acknowledgments ###

* Based on IDL and Python code by Jennifer Lotz and Greg Snyder.

### Citing ###

* If you use this code for scientific publication, please cite
the package using its Zenodo record:

T.B.D.

* Additionally, if you use the Gini and M20 statistics, you should cite:
  * Abraham R. G., van den Bergh S., Nair P., 2003, ApJ, 588, 218
  * Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163.

* If you use the concentration, asymmetry and clumpiness (CAS) statistics,
you should cite:
  * Bershady M. A., Jangren A., Conselice C. J., 2000, AJ, 119, 2645
  * Conselice C. J., 2003, ApJS, 147, 1

* If you use the multimode, intensity and deviation (MID) statistics,
you should cite:
  * Freeman P. E., Izbicki R., Lee A. B., Newman J. A., Conselice C. J.,
    Koekemoer A. M., Lotz J. M., Mozena M., 2013, MNRAS, 434, 282

* If you use the shape asymmetry statistic, you should cite:
  * Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

The list of references above is provided as a starting point. Please see
the references for further information.

### Disclaimer ###

This package is not meant to be the "official" implementation of any
of the morphological statistics described above. Please contact the
authors of the original papers for a reference implementation.
Also see the LICENSE.

### Licensing ###

* Licensed under a 3-Clause BSD License.
