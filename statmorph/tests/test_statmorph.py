"""
Tests for the statmorph morphology code. Based on the tutorial example.
"""
# Author: Vicente Rodriguez-Gomez <vrg@jhu.edu>
# Licensed under a 3-Clause BSD License.
from __future__ import absolute_import, division, print_function

import numpy as np
import time
import scipy.ndimage as ndi
from astropy.modeling import models
import photutils
import statmorph

__all__ = ['runall']

tutorial_values = {
    'xc_centroid':120.438308743,
    'yc_centroid':96.4583177325,
    'ellipticity_centroid':0.490187141932,
    'elongation_centroid':1.96150407777,
    'orientation_centroid':0.504352692498,
    'xc_asymmetry':120.14904555,
    'yc_asymmetry':96.0958951908,
    'ellipticity_asymmetry':0.490238908643,
    'elongation_asymmetry':1.96170327033,
    'orientation_asymmetry':0.504687057394,
    'rpetro_circ':33.00363072517078,
    'rpetro_ellip':44.95218238959798,
    'rhalf_circ':14.760350010415168,
    'rhalf_ellip':19.338081776233196,
    'r20':6.289384232990396,
    'r80':26.55128424232843,
    'gini':0.535076026307,
    'm20':-1.90753710158,
    'gini_m20_bulge':0.120549541616,
    'sn_per_pixel':6.66941658013,
    'concentration':3.12738702225,
    'asymmetry':-0.0161047547922,
    'smoothness':0.0430059956785,
    'sersic_amplitude':1.04307316369,
    'sersic_rhalf':19.551686099,
    'sersic_n':1.45596636058,
    'sersic_xc':119.994694189,
    'sersic_yc':95.9941893731,
    'sersic_ellip':0.500116075037,
    'sersic_theta':0.500753943935,
    'sky_mean':0.00795852764083,
    'sky_median':0.00834031437676,
    'sky_sigma':0.0403136809839,
}

def tutorial_test():
    """
    Simple test based on the notebook tutorial.
    """
    # Create model galaxy
    ny, nx = 240, 240
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = models.Sersic2D(amplitude=1, r_eff=20, n=1.5, x_0=0.5*nx, y_0=0.4*ny,
                                   ellip=0.5, theta=0.5)
    image = sersic_model(x, y)

    # Convolve with PSF
    size = 20  # on each side from the center
    sigma_psf = 2.0
    y, x = np.mgrid[-size:size+1, -size:size+1]
    psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
    psf /= np.sum(psf)
    image = ndi.convolve(image, psf)

    # Add noise
    np.random.seed(0)
    snp = 25.0
    image += (1.0 / snp) * np.random.standard_normal(size=(ny, nx))

    # Define weight map
    gain = 100.0

    # Create segmentation map
    threshold = photutils.detect_threshold(image, snr=1.5)
    npixels = 5  # minimum number of connected pixels
    segm = photutils.detect_sources(image, threshold, npixels)
    # Keep only the largest segment (label=0 is reserved for the background)
    label = np.argmax(segm.areas[1:]) + 1
    segmap = segm.data == label
    # Regularize a bit the shape of the segmentation map
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    # Measure morphology
    start = time.time()
    print('Running statmorph test...')
    source_morphs = statmorph.source_morphology(image, segmap, gain=gain, psf=psf)
    morph = source_morphs[0]
    print('Time: %g s.' % (time.time() - start))

    # Check results
    assert morph['flag'] == 0
    assert morph['flag_sersic'] == 0
    for key, value in tutorial_values.items():
        value0 = tutorial_values[key]
        relative_error = np.abs((value - value0) / value0)
        assert relative_error < 1e-6

def runall():
    """
    Run all tests.
    """
    tutorial_test()
    print('All tests finished successfully.')
