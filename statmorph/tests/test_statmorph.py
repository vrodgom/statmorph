"""
Tests for the statmorph morphology package.
"""
# Author: Vicente Rodriguez-Gomez <vrodgom.astro@gmail.com>
# Licensed under a 3-Clause BSD License.
import numpy as np
import os
import pytest
import statmorph
from astropy.modeling import models
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

__all__ = ['runall']


def test_quantile():
    from statmorph.statmorph import _quantile
    quantiles = np.linspace(0, 1, 11)
    data = np.arange(25, dtype=np.float64)**2
    # Compare with np.percentile (note that _quantile() assumes that the
    # input array is already sorted, so it's much faster in these cases).
    res1 = []; res2 = []
    for q in quantiles:
        res1.append(_quantile(data, q))
        res2.append(np.percentile(data, 100*q, interpolation='lower'))
    assert_allclose(res1, res2)
    # Check out-of-range input.
    with pytest.raises(ValueError):
        _ = _quantile(data, -0.5)
        _ = _quantile(data, 1.5)


def test_convolved_sersic():
    from scipy.signal import fftconvolve
    # Create Gaussian PSF.
    size = 10  # on each side from the center
    sigma_psf = 2.0
    y, x = np.mgrid[-size:size + 1, -size:size + 1]
    psf = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma_psf ** 2))
    psf /= np.sum(psf)
    # Create 2D Sersic profile.
    ny, nx = 25, 25
    y, x = np.mgrid[0:ny, 0:nx]
    sersic = models.Sersic2D(amplitude=1, r_eff=5, n=1.5, x_0=12, y_0=12,
                             ellip=0.5, theta=0)
    z = sersic(x, y)
    # Create "convolved" Sersic profile with same properties as normal one.
    convolved_sersic = statmorph.ConvolvedSersic2D(
        amplitude=1, r_eff=5, n=1.5, x_0=12, y_0=12, ellip=0.5, theta=0)
    with pytest.raises(AssertionError):
        _ = convolved_sersic(x, y)  # PSF not set yet
    convolved_sersic.set_psf(psf)
    z_convolved = convolved_sersic(x, y)
    # Compare results.
    assert_allclose(z_convolved, fftconvolve(z, psf, mode='same'))


def test_missing_arguments():
    label = 1
    image = np.ones((3, 3), dtype=np.float64)
    segmap = np.ones((3, 3), dtype=np.int64)
    with pytest.raises(AssertionError):
        _ = statmorph.SourceMorphology(image, segmap, label)


def test_catastrophic():
    label = 1
    image = np.full((3, 3), -1.0, dtype=np.float64)
    segmap = np.full((3, 3), label, dtype=np.int64)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0)
    assert len(w) == 1
    assert w[0].category == AstropyUserWarning
    assert 'Total flux is nonpositive.' in str(w[0].message)
    assert morph.flag == 4


def test_masked_centroid():
    label = 1
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-(x - 5) ** 2 - (y - 5) ** 2)
    segmap = np.int64(image > 1e-3)
    mask = np.zeros((ny, nx), dtype=np.bool_)
    mask[5, 5] = True
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           mask=mask)
    assert w[0].category == AstropyUserWarning
    assert 'Centroid is masked.' in str(w[0].message)
    assert morph.flag == 2


def test_bright_pixel():
    """
    Test bright pixel outside of main segment. Note that
    we do not remove outliers.
    """
    label = 1
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-(x - 5) ** 2 - (y - 5) ** 2)
    image[7, 7] = 1.0
    segmap = np.int64(image > 1e-3)
    segmap[5, 5] = 0
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           n_sigma_outlier=-1)
    assert w[0].category == AstropyUserWarning
    assert 'Adding brightest pixel to segmap.' in str(w[0].message)
    assert morph.flag == 2


def test_negative_source():
    label = 1
    ny, nx = 51, 51
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((x-nx//2)**2 + (y-ny//2)**2)
    image = np.ones((ny, nx), dtype=np.float64)
    locs = r > 0
    image[locs] = 2.0/r[locs] - 1.0
    segmap = np.int64(r < 2)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0)
    assert w[0].category == AstropyUserWarning
    assert 'Total flux sum is negative.' in str(w[0].message)
    assert morph.flag == 2


def test_tiny_source():
    """
    Test tiny source (actually consisting of a single bright pixel).
    Note that we do not remove outliers.
    """
    label = 1
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 1.0
    segmap = np.int64(image)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           n_sigma_outlier=-1)
    assert w[0].category == AstropyUserWarning
    assert 'Nonpositive second moment.' in str(w[0].message)
    assert morph.flag == 2


def test_insufficient_data():
    """
    Test insufficient data for Sersic fit (< 7 pixels).
    Note that we do not remove outliers.
    """
    label = 1
    image = np.zeros((2, 3), dtype=np.float64)
    image[:, 1] = 1.0
    segmap = np.int64(image)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           n_sigma_outlier=-1)
    assert w[-2].category == AstropyUserWarning
    assert '[sersic] Not enough data for fit.' in str(w[-2].message)
    assert morph.flag == 2


def test_asymmetric():
    """
    Test a case in which the asymmetry center is pushed outside of
    the image boundaries.
    """
    label = 1
    y, x = np.mgrid[0:25, 0:25]
    image = x - 20
    segmap = np.int64(image > 0)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0)
    assert w[0].category == AstropyUserWarning
    assert 'Minimizer tried to exit bounds.' in str(w[0].message)
    assert morph.flag == 2
    assert morph._use_centroid


def test_small_source():
    np.random.seed(1)
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-(x - 5) ** 2 - (y - 5) ** 2)
    image += 0.001 * np.random.standard_normal(size=(ny, nx))
    segmap = np.int64(image > 0.1)
    label = 1
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           verbose=True)
    assert w[-1].category == AstropyUserWarning
    assert 'Single clump!' in str(w[-1].message)
    assert morph.flag == 0
    assert morph.multimode == 0
    assert morph.intensity == 0


def test_full_segmap():
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-(x - 5) ** 2 - (y - 5) ** 2)
    segmap = np.ones((ny, nx), dtype=np.int64)
    label = 1
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0,
                                           verbose=True)
    assert w[-1].category == AstropyUserWarning
    assert 'Image is not background-subtracted.' in str(w[-1].message)
    assert morph.flag == 2
    assert morph._slice_skybox == (slice(0, 0), slice(0, 0))


def test_random_noise():
    np.random.seed(1)
    ny, nx = 11, 11
    image = 0.1 * np.random.standard_normal(size=(ny, nx))
    weightmap = 0.01 * np.random.standard_normal(size=(ny, nx))
    segmap = np.ones((ny, nx), dtype=np.int64)
    label = 1
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label,
                                           weightmap=weightmap)
    assert w[-1].category == AstropyUserWarning
    assert morph.flag == 2


def test_empty_gini_segmap():
    """
    This pathological case results in an "empty" Gini segmap.
    """
    label = 1
    np.random.seed(0)
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = x - 9.0
    segmap = np.int64(image > 0)
    image += 0.1 * np.random.standard_normal(size=(ny, nx))
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0)
    assert w[-1].category == AstropyUserWarning
    assert 'Segmaps are empty!' in str(w[-1].message)
    assert morph.flag == 2


def test_full_gini_segmap():
    """
    This produces a "full" Gini segmap.
    """
    label = 1
    ny, nx = 11, 11
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-((x - nx // 2) ** 2 + (y - ny // 2) ** 2) / 50)
    segmap = np.int64(image > 0.5)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(image, segmap, label, gain=1.0)
    assert w[-3].category == AstropyUserWarning
    assert 'Full Gini segmap!' in str(w[-3].message)
    assert morph.flag == 2


def test_merger():
    """
    Test a "merger" scenario. This manages to produce different Gini
    and MID segmaps, as well as a failed Sersic fit.
    """
    label = 1
    ny, nx = 25, 25
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.exp(-(x-8)**2/4 - (y-12)**2)
    image += np.exp(-(x-16)**2/4 - (y-12)**2)
    segmap = np.int64(np.abs(image) > 1e-3)
    with pytest.warns() as w:
        morph = statmorph.SourceMorphology(
            image, segmap, label, gain=1.0, verbose=True)
    assert w[-1].category == AstropyUserWarning
    assert 'Gini and MID segmaps are quite different.' in str(w[-1].message)
    assert morph.flag == 1


class TestSourceMorphology(object):
    """
    Check measurements for a test galaxy image + segmap + mask.
    """
    def setup_class(self):
        self.correct_values = {
            'xc_centroid': 81.66578971349672,
            'yc_centroid': 80.53461797343211,
            'ellipticity_centroid': 0.04855709579576,
            'elongation_centroid': 1.05103521775316,
            'orientation_centroid': -0.85797602066195,
            'xc_asymmetry': 82.23985214801982,
            'yc_asymmetry': 80.76076242700849,
            'ellipticity_asymmetry': 0.04806946962244,
            'elongation_asymmetry': 1.05049682522881,
            'orientation_asymmetry': -0.85405676626920,
            'flux_circ': 5758942.65115976985544,
            'flux_ellip': 5758313.01348320022225,
            'rpetro_circ': 40.93755531944313,
            'rpetro_ellip': 41.64283484446126,
            'rmax_circ': 54.10691995800065,
            'rmax_ellip': 54.57312319389109,
            'rhalf_circ': 21.60803205322342,
            'rhalf_ellip': 22.08125638365687,
            'r20': 11.69548630967248,
            'r50': 21.62164455681452,
            'r80': 32.07883340820674,
            'gini': 0.38993180299621,
            'm20': -1.54448930789228,
            'gini_m20_bulge': -0.95950648479940,
            'gini_m20_merger': -0.15565152883078,
            'sn_per_pixel': 6.80319166183472,
            'concentration': 2.19100140632153,
            'asymmetry': 0.00377345808887,
            'smoothness': 0.00430880839402,
            'multimode': 0.23423423423423,
            'intensity': 0.51203949030140,
            'deviation': 0.01522525597953,
            'outer_asymmetry': -0.01821399684443,
            'shape_asymmetry': 0.16308278287864,
            'sersic_amplitude': 1296.95288208155739,
            'sersic_rhalf': 22.45788866502031,
            'sersic_n': 0.61206828194077,
            'sersic_xc': 81.56197595338546,
            'sersic_yc': 80.40465135599014,
            'sersic_ellip': 0.05083866217150,
            'sersic_theta': 2.47831542907976,
            'sky_mean': 3.48760604858398,
            'sky_median': -2.68543863296509,
            'sky_sigma': 150.91754150390625,
            'xmin_stamp': 0,
            'ymin_stamp': 0,
            'xmax_stamp': 161,
            'ymax_stamp': 161,
            'nx_stamp': 162,
            'ny_stamp': 162,
            'flag': 0,
            'flag_sersic': 0,
        }

        # Run statmorph on the same galaxy from which the above values
        # were obtained.
        curdir = os.path.dirname(__file__)
        with fits.open('%s/data/slice.fits' % (curdir,)) as hdulist:
            self.image = hdulist[0].data
            self.segmap = hdulist[1].data
            self.mask = np.bool_(hdulist[2].data)
        self.gain = 1.0

    def test_no_psf(self, print_values=False):
        source_morphs = statmorph.source_morphology(
            self.image, self.segmap, mask=self.mask, gain=self.gain)
        morph = source_morphs[0]
        for key in self.correct_values:
            assert_allclose(morph[key], self.correct_values[key],
                            err_msg="%s value did not match." % (key,))
            if print_values:
                print("'%s': %.14g," % (key, morph[key]))

    def test_psf(self):
        # Try delta-like PSF, which should give the same results as no PSF.
        psf = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.float64)
        source_morphs = statmorph.source_morphology(
            self.image, self.segmap, mask=self.mask, gain=self.gain, psf=psf)
        morph = source_morphs[0]
        for key in self.correct_values:
            assert_allclose(morph[key], self.correct_values[key],
                            err_msg="%s value did not match." % (key,))

    def test_weightmap(self):
        # Manually create weight map instead of using the gain argument.
        weightmap = np.sqrt(
            np.abs(self.image) / self.gain + self.correct_values['sky_sigma']**2)
        source_morphs = statmorph.source_morphology(
            self.image, self.segmap, mask=self.mask, weightmap=weightmap)
        morph = source_morphs[0]
        for key in self.correct_values:
            assert_allclose(morph[key], self.correct_values[key],
                            err_msg="%s value did not match." % (key,))


def runall(print_values=False):
    """
    Run the most basic tests. Keep this function for backward compatibility.
    """
    test = TestSourceMorphology()
    test.setup_class()
    test.test_no_psf(print_values=print_values)
