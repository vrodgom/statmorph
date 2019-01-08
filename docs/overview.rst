
Overview
============

For a given (background-subtracted) image and a corresponding segmentation map
indicating the source(s) of interest, statmorph calculates the following
morphological statistics for each source:

- Gini-M20 statistics (Lotz et al. 2004; Snyder et al. 2015a,b)
- Concentration, Asymmetry and Smoothness (CAS) statistics
  (Bershady et al. 2000; Conselice 2003; Lotz et al. 2004)
- Multimode, Intensity and Deviation (MID) statistics (Freeman et al. 2013;
  Peth et al. 2016)
- Outer asymmetry and shape asymmetry (Wen et al. 2014; Pawlik et al. 2016)
- Sérsic index (Sérsic 1968)
- Several shape and size measurements associated to the above statistics
  (ellipticity, Petrosian radius, half-light radius, etc.)

.. ~ For more information, please see:

.. ~ - `Rodriguez-Gomez et al. (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_

The current Python implementation is largely based on IDL and Python code
originally written by Jennifer Lotz and Greg Snyder.

**Authors**

- Vicente Rodriguez-Gomez (v.rodriguez [at] irya.unam.mx)
- Jennifer Lotz
- Greg Snyder

**Acknowledgments**

- We thank Peter Freeman and Mike Peth for sharing their IDL
  implementation of the MID statistics.

**Citing**

If you use this code for a scientific publication, please cite the following
article:

- `Rodriguez-Gomez et al. (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_

Optionally, the Python package can also be cited using its Zenodo record:

.. image:: https://zenodo.org/badge/95412529.svg
   :target: https://zenodo.org/badge/latestdoi/95412529

.. ~ Finally, below we provide some of the main references that introduce the
.. ~ morphological parameters implemented in this code. The following list is
.. ~ provided as a starting point and is not meant to be exhaustive. Please
.. ~ see consult each publication for more information.

.. ~ - Gini--M20 statistics:

.. ~   - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163
.. ~   - Snyder G. F. et al., 2015, MNRAS, 451, 4290
.. ~   - Snyder G. F. et al., 2015, MNRAS, 454, 1886

.. ~ - Concentration, asymmetry and clumpiness (CAS) statistics:

.. ~   - Bershady M. A., Jangren A., Conselice C. J., 2000, AJ, 119, 2645
.. ~   - Conselice C. J., 2003, ApJS, 147, 1
.. ~   - Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

.. ~ - Multimode, intensity and deviation (MID) statistics:

.. ~   - Freeman P. E., Izbicki R., Lee A. B., Newman J. A., Conselice C. J.,
.. ~     Koekemoer A. M., Lotz J. M., Mozena M., 2013, MNRAS, 434, 282
.. ~   - Peth M. A. et al., 2016, MNRAS, 458, 963

.. ~ - Outer asymmetry and shape asymmetry:

.. ~   - Wen Z. Z., Zheng X. Z., Xia An F., 2014, ApJ, 787, 130
.. ~   - Pawlik M. M., Wild V., Walcher C. J., Johansson P. H., Villforth C.,
.. ~     Rowlands K., Mendez-Abreu J., Hewlett T., 2016, MNRAS, 456, 3032

.. ~ - Sérsic index:

.. ~   - Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico
.. ~     de Córdoba, Córdoba, Argentina

**Disclaimer**

This package is not meant to be the "official" implementation of any
of the morphological statistics listed above. Please contact the
authors of the original publications for a "reference" implementation.

**Licensing**

Licensed under a 3-Clause BSD License.
