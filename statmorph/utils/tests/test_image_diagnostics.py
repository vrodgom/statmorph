"""
Some trivial tests for the ``image_diagnostics`` plotting utility.
Usually these are skipped, since Matplotlib is not listed in the
statmorph requirements.
"""
# Author: Vicente Rodriguez-Gomez <vrodgom.astro@gmail.com>
# Licensed under a 3-Clause BSD License.
import numpy as np
import os
import pytest
import statmorph
from astropy.io import fits
try:
    import matplotlib
    from statmorph.utils import make_figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@pytest.mark.skipif('not HAS_MATPLOTLIB', reason='Requires Matplotlib.')
def test_invalid_input():
    with pytest.raises(TypeError):
        make_figure('foo')

@pytest.mark.skipif('not HAS_MATPLOTLIB', reason='Requires Matplotlib.')
def test_make_figure():
    curdir = os.path.dirname(__file__)
    hdulist = fits.open('%s/../../tests/data/slice.fits' % (curdir,))
    image = hdulist[0].data
    segmap = hdulist[1].data
    mask = np.bool8(hdulist[2].data)
    gain = 1.0
    source_morphs = statmorph.source_morphology(image, segmap, mask=mask, gain=gain)
    morph = source_morphs[0]
    fig = make_figure(morph)
    assert isinstance(fig, matplotlib.figure.Figure)