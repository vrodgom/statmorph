"""
Tests for the statmorph morphology code. Based on the tutorial example.
"""
# Author: Vicente Rodriguez-Gomez <v.rodriguez@irya.unam.mx>
# Licensed under a 3-Clause BSD License.
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import time
import pytest
import statmorph
from astropy.io import fits

__all__ = ['runall']

correct_values = {
    'xc_centroid':81.66578971349672,
    'yc_centroid':80.53461797343211,
    'ellipticity_centroid':0.04855709579576,
    'elongation_centroid':1.05103521775316,
    'orientation_centroid':-0.85797602066195,
    'xc_asymmetry':82.23985214801982,
    'yc_asymmetry':80.76076242700849,
    'ellipticity_asymmetry':0.04806946962244,
    'elongation_asymmetry':1.05049682522881,
    'orientation_asymmetry':-0.85405676626920,
    'rpetro_circ':40.93755531944313,
    'rpetro_ellip':41.64283484446126,
    'rhalf_circ':21.60803205322342,
    'rhalf_ellip':22.08125638365687,
    'r20':11.69548630967248,
    'r80':32.07883340820674,
    'gini':0.38993180299621,
    'm20':-1.54448930789228,
    'sn_per_pixel':6.80319166183472,
    'concentration':2.19100140632153,
    'asymmetry':0.00377345808887,
    'smoothness':0.00430880839402,
    'multimode':0.23423423423423,
    'intensity':0.51203949030140,
    'deviation':0.01522525597953,
    'outer_asymmetry':-0.01821399684443,
    'shape_asymmetry':0.16308278287864,
    'sersic_amplitude':1296.95288208155739,
    'sersic_rhalf':22.45788866502031,
    'sersic_n':0.61206828194077,
    'sersic_xc':81.56197595338546,
    'sersic_yc':80.40465135599014,
    'sersic_ellip':0.05083866217150,
    'sersic_theta':2.47831542907976,
    'sky_mean':3.48760604858398,
    'sky_median':-2.68543863296509,
    'sky_sigma':150.91754150390625,
}

def test_morph():
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
    test_morph()
    print('Time: %g s.' % (time.time() - start))
    print('All tests finished successfully.')
