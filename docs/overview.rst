
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
- Single and double Sérsic indices (Sérsic 1968)
- Several shape and size measurements associated to the above statistics
  (ellipticity, Petrosian radius, half-light radius, etc.)

.. ~ For more information, please see:

.. ~ - `Rodriguez-Gomez et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_

The current Python implementation is largely based on IDL and Python code
originally written by Jennifer Lotz and Greg Snyder.

**Authors**

- Vicente Rodriguez-Gomez (vrodgom.astro@gmail.com)
- Jennifer Lotz
- Greg Snyder

**Acknowledgments**

- We thank Peter Freeman and Mike Peth for sharing their IDL
  implementation of the MID statistics.

**Citing**

If you use this code for a scientific publication, please cite the following
article:

- `Rodriguez-Gomez et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_

.. ~ Optionally, the Python package can also be cited using its Zenodo record:

.. ~ .. image:: https://zenodo.org/badge/95412529.svg
.. ~    :target: https://zenodo.org/badge/latestdoi/95412529

**Disclaimer**

This package is not meant to be the "official" implementation of any
of the morphological statistics listed above. Please contact the
authors of the original publications for a "reference" implementation.

**Licensing**

Licensed under a 3-Clause BSD License.
