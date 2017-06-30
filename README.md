# statmorph

Python code for calculating non-parametric morphological parameters,
following the methodology of Lotz et al. (2004).

### Brief description ###

T.B.D. (brief description of Gini, M20, etc.)

### Further information ###

Further details can be found in
[Lotz et al. (2004)](http://adsabs.harvard.edu/abs/2004AJ....128..163L).

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
