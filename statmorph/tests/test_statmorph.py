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
    'yc_centroid':81.03461797343211,
    'ellipticity_centroid':0.04855709579576,
    'elongation_centroid':1.05103521775316,
    'orientation_centroid':-0.85797602066195,
    'xc_asymmetry':82.23816149429084,
    'yc_asymmetry':80.76122537398356,
    'ellipticity_asymmetry':0.04862719043001,
    'elongation_asymmetry':1.05111265525025,
    'orientation_asymmetry':-0.85900294920303,
    'rpetro_circ':40.93685686885102,
    'rpetro_ellip':41.65687159352140,
    'rhalf_circ':21.60432043808139,
    'rhalf_ellip':22.09299918129802,
    'r20':11.69540955266178,
    'r80':32.07888648441381,
    'gini':0.39017652998146,
    'm20':-1.54294057566943,
    'sn_per_pixel':6.79978752136230,
    'concentration':2.19101925047364,
    'asymmetry':0.00216882009677,
    'smoothness':0.00430980802075,
    'multimode':0.23423423423423,
    'intensity':0.51203949030140,
    'deviation':0.01522525597953,
    'outer_asymmetry':-0.01939061100961,
    'shape_asymmetry':0.16270478667804,
    'sersic_amplitude':1296.95301375643112,
    'sersic_rhalf':22.45788881197571,
    'sersic_n':0.61206811649601,
    'sersic_xc':82.06197530371749,
    'sersic_yc':80.90465047895617,
    'sersic_ellip':0.05083868323162,
    'sersic_theta':2.47831539294638,
    'sky_mean':3.48760604858398,
    'sky_median':-2.68543863296509,
    'sky_sigma':150.91754150390625,
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

    #for key in correct_values:
    #    print("'%s':%.14f," % (key, morph[key]))

    # Check results
    assert morph['flag'] == 0
    assert morph['flag_sersic'] == 0
    for key in correct_values:
        value = morph[key]
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
