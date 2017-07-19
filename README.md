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

* numpy
* scipy
* scikit-image
* astropy
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
hdulist_stmask = fits.open('stmask.fits.gz')
hdulist_weights = fits.open('weights.fits.gz')

image = hdulist_image['PRIMARY'].data
segmap = hdulist_segmap['PRIMARY'].data
mask = np.asarray(hdulist_stmask['PRIMARY'].data, dtype=np.bool8)
variance = hdulist_weights['PRIMARY'].data

source_morphology = statmorph.source_morphology(
    image, segmap, mask=mask, variance=variance)

hdulist_image.close()
hdulist_segmap.close()
hdulist_stmask.close()
hdulist_weights.close()

# Print some properties of the first source in the segmentation map
morph = source_morphology[0]
quantities = [
    'petrosian_radius_circ',
    'petrosian_radius_ellip',
    'gini',
    'm20',
    'sn_per_pixel',
    'asymmetry',
    'concentration',
    'smoothness',
    'multimode',
    'intensity',
    'deviation',
    'rmax',
    'outer_asymmetry',
    'shape_asymmetry',
]

start_all = time.time()
for quantity in quantities:
    start = time.time()
    value = morph[quantity]
    print('%25s: %10.6f   (Time: %.6f s)' % (
          quantity, value, time.time() - start))
print('\nTotal time: %.6f s.' % (time.time() - start_all))

```

For Pan-STARRS galaxy **J235958.6+281704** in the g-band, this returns:

```
    petrosian_radius_circ:  55.290919   (Time: 0.029993 s)
   petrosian_radius_ellip:  97.719538   (Time: 0.797421 s)
                     gini:   0.574926   (Time: 0.138750 s)
                      m20:  -1.957585   (Time: 0.029239 s)
             sn_per_pixel:   3.971230   (Time: 0.001377 s)
                asymmetry:   0.147391   (Time: 0.117764 s)
            concentration:   3.204953   (Time: 0.009796 s)
               smoothness:   0.079202   (Time: 0.009536 s)
                multimode:   0.027788   (Time: 1.078006 s)
                intensity:   0.018720   (Time: 0.139468 s)
                deviation:   0.018686   (Time: 0.002891 s)
                     rmax: 119.067208   (Time: 0.301612 s)
          outer_asymmetry:   0.191176   (Time: 0.332309 s)
          shape_asymmetry:   0.198903   (Time: 0.004798 s)

Total time: 2.993595 s.
```

### Authors ###
* Vicente Rodriguez-Gomez (vrg [at] jhu.edu)

### Contributors ###
* T.B.D.

### Acknowledgments ###

* Based on IDL and Python code by Jennifer Lotz, Greg Snyder, Peter
  Freeman and Mike Peth.

### Citing ###

If you use this code for scientific publication, please cite
the package using its Zenodo record:

* T.B.D.

In addition, below we provide some of the main references that should
be cited when using each of the morphological parameters. This list is
provided as a starting point and is not meant to be exhaustive. Please
see the references within each publication for a more complete list.

* Gini--M20 statistics:
  * Abraham R. G., van den Bergh S., Nair P., 2003, ApJ, 588, 218
  * Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163.

* Concentration, asymmetry and clumpiness (CAS) statistics:
  * Bershady M. A., Jangren A., Conselice C. J., 2000, AJ, 119, 2645
  * Conselice C. J., 2003, ApJS, 147, 1

* Multimode, intensity and deviation (MID) statistics:
  * Freeman P. E., Izbicki R., Lee A. B., Newman J. A., Conselice C. J.,
    Koekemoer A. M., Lotz J. M., Mozena M., 2013, MNRAS, 434, 282
  * Peth M. A. et al., 2016, MNRAS, 458, 963

* Outer asymmetry:
  * Wen Z. Z., Zheng X. Z., Xia An F., 2014, ApJ, 787, 130
  * Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

* Shape asymmetry:
  * Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
    Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

### Disclaimer ###

This package is not meant to be the "official" implementation of any
of the morphological statistics described above. Please contact the
authors of the original publications for a "reference" implementation.
Also see the LICENSE.

### Licensing ###

* Licensed under a 3-Clause BSD License.
