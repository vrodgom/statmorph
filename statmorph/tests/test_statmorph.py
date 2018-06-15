"""
Tests for the statmorph morphology code. Based on the tutorial example.
"""
# Author: Vicente Rodriguez-Gomez <vrg@jhu.edu>
# Licensed under a 3-Clause BSD License.
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import time
import statmorph
from astropy.io import fits

__all__ = ['runall']

correct_values = {
    'xc_centroid':82.16578971349672,
    'yc_centroid':81.0346179734321,
    'ellipticity_centroid':0.04855709579575873,
    'elongation_centroid':1.0510352177531561,
    'orientation_centroid':-0.8579760206619491,
    'xc_asymmetry':82.23816149429084,
    'yc_asymmetry':80.76122537398356,
    'ellipticity_asymmetry':0.04862719043000929,
    'elongation_asymmetry':1.0511126552502466,
    'orientation_asymmetry':-0.8590029492030308,
    'rpetro_circ':40.93685686885102,
    'rpetro_ellip':41.6568715935214,
    'rhalf_circ':21.60432043808139,
    'rhalf_ellip':22.092999181298023,
    'r20':11.695409552661776,
    'r80':32.07888648441381,
    'gini':0.3901765299814595,
    'm20':-1.5429405756694308,
    'gini_m20_bulge':-0.8493683576528599,
    'sn_per_pixel':6.7997875,
    'concentration':2.191019250473638,
    'asymmetry':0.002168820096766045,
    'smoothness':0.019515863990804592,
    'sersic_amplitude':1296.9528812407366,
    'sersic_rhalf':22.45788866708879,
    'sersic_n':0.6120682828213252,
    'sersic_xc':81.561975952764,
    'sersic_yc':80.4046513562675,
    'sersic_ellip':0.05083866210052884,
    'sersic_theta':2.4783154293185397,
    'sky_mean':3.487606,
    'sky_median':-2.6854386,
    'sky_sigma':150.91754,
}

def test1():
    """
    Check values for a randomly chosen galaxy.
    """
    curdir = os.path.dirname(__file__)
    hdulist = fits.open('%s/data_slice.fits' % (curdir))
    image = hdulist[0].data
    segmap = hdulist[1].data
    mask = np.bool8(hdulist[2].data)
    gain = 1.0
    source_morphs = statmorph.source_morphology(image, segmap, mask=mask, gain=gain)
    morph = source_morphs[0]

    # Check results
    assert morph['flag'] == 0
    assert morph['flag_sersic'] == 0
    for key, value in correct_values.items():
        value0 = correct_values[key]
        relative_error = np.abs((value - value0) / value0)
        assert relative_error < 1e-6

def runall():
    """
    Run all tests.
    """
    start = time.time()
    print('Running statmorph tests...')
    test1()
    print('Time: %g s.' % (time.time() - start))
    print('All tests finished successfully.')
