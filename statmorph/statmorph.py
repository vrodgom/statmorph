"""
Python code for calculating non-parametric morphological diagnostics
of galaxy images.
"""
# Author: Vicente Rodriguez-Gomez <vrg@jhu.edu>
# Licensed under a 3-Clause BSD License.
from __future__ import absolute_import, division, print_function

import warnings
import time
import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndi
import scipy.signal
import skimage.measure
import skimage.transform
import skimage.feature
import skimage.morphology
from astropy.utils import lazyproperty
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
import photutils

__all__ = ['ConvolvedSersic2D', 'SourceMorphology', 'source_morphology',
           '__version__']

__version__ = '0.3.2'

def _quantile(sorted_values, q):
    """
    For a sorted (in increasing order) 1-d array, return the
    value corresponding to the quantile ``q``.
    """
    assert ((q >= 0) & (q <= 1))
    if q == 1:
        return sorted_values[-1]
    else:
        return sorted_values[int(q*len(sorted_values))]

def _mode(a, axis=None):
    """
    Takes a masked array as input and returns the "mode"
    as defined in Bertin & Arnouts (1996):
    mode = 2.5 * median - 1.5 * mean
    """
    return 2.5*np.ma.median(a, axis=axis) - 1.5*np.ma.mean(a, axis=axis)

def _local_variance(image):
    """
    Calculate a map of the local variance around each pixel, based on
    its 8 adjacent neighbors

    Notes
    -----
    ndi.generic_filter(image, np.std, ...) is too slow,
    so we do a workaround using ndi.convolve.
    """
    # Pixel weights, excluding central pixel.
    w = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]], dtype=np.float64)
    w = w / np.sum(w)
    
    # Use the fact that var(x) = <x^2> - <x>^2.
    local_mean = ndi.convolve(image, w)
    local_mean2 = ndi.convolve(image**2, w)
    local_var = local_mean2 - local_mean**2
    
    return local_var

def _aperture_area(ap, mask, **kwargs):
    """
    Calculate the area of a photutils aperture object,
    excluding masked pixels.
    """
    return ap.do_photometry(np.float64(~mask), **kwargs)[0][0]

def _aperture_mean_nomask(ap, image, **kwargs):
    """
    Calculate the mean flux of an image for a given photutils
    aperture object. Note that we do not use ``_aperture_area``
    here. Instead, we divide by the full area of the
    aperture, regardless of masked and out-of-range pixels.
    This avoids problems when the aperture is larger than the
    region of interest.
    """
    return ap.do_photometry(image, **kwargs)[0][0] / ap.area()

def _fraction_of_total_function_circ(r, image, center, fraction, total_sum):
    """
    Helper function to calculate ``_radius_at_fraction_of_total_circ``.
    """
    if (r < 0) | (fraction < 0) | (fraction > 1) | (total_sum <= 0):
        raise AssertionError
    elif r == 0:
        cur_fraction = 0.0
    else:
        ap = photutils.CircularAperture(center, r)
        # Force flux sum to be positive:
        ap_sum = np.abs(ap.do_photometry(image, method='exact')[0][0])
        cur_fraction = ap_sum / total_sum

    return cur_fraction - fraction

def _radius_at_fraction_of_total_circ(image, center, r_total, fraction):
    """
    Return the radius (in pixels) of a concentric circle that
    contains a given fraction of the light within ``r_total``.
    """
    flag = 0  # flag=1 indicates a problem

    ap_total = photutils.CircularAperture(center, r_total)

    total_sum = ap_total.do_photometry(image, method='exact')[0][0]
    if total_sum == 0:
        raise AssertionError
    elif total_sum < 0:
        warnings.warn('Total flux sum is negative.', AstropyUserWarning)
        flag = 1
        total_sum = np.abs(total_sum)

    # Find appropriate range for root finder
    npoints = 100
    r_grid = np.linspace(0.0, r_total, num=npoints)
    i = 0  # initial value
    while True:
        if i >= npoints:
            raise Exception('Root not found within range.')
        r = r_grid[i]
        curval = _fraction_of_total_function_circ(
            r, image, center, fraction, total_sum)
        if curval == 0:
            warnings.warn('Found root by pure chance!', AstropyUserWarning)
            return r, flag
        elif curval < 0:
            r_min = r
        elif curval > 0:
            r_max = r
            break
        i += 1

    r = opt.brentq(_fraction_of_total_function_circ, r_min, r_max,
                   args=(image, center, fraction, total_sum), xtol=1e-6)

    return r, flag

def _fraction_of_total_function_ellip(a, image, center, elongation, theta,
                                      fraction, total_sum):
    """
    Helper function to calculate ``_radius_at_fraction_of_total_ellip``.
    """
    if (a < 0) | (fraction < 0) | (fraction > 1) | (total_sum <= 0):
        raise AssertionError
    elif a == 0:
        cur_fraction = 0.0
    else:
        b = a / elongation
        ap = photutils.EllipticalAperture(center, a, b, theta)
        # Force flux sum to be positive:
        ap_sum = np.abs(ap.do_photometry(image, method='exact')[0][0])
        cur_fraction = ap_sum / total_sum
    
    return cur_fraction - fraction

def _radius_at_fraction_of_total_ellip(image, center, elongation, theta,
                                       a_total, fraction):
    """
    Return the semimajor axis (in pixels) of a concentric ellipse that
    contains a given fraction of the light within a larger ellipse of
    semimajor axis ``a_total``.
    """
    flag = 0  # flag=1 indicates a problem

    b_total = a_total / elongation
    ap_total = photutils.EllipticalAperture(center, a_total, b_total, theta)

    total_sum = ap_total.do_photometry(image, method='exact')[0][0]
    if total_sum == 0:
        raise AssertionError
    elif total_sum < 0:
        warnings.warn('Total flux sum is negative.', AstropyUserWarning)
        flag = 1
        total_sum = np.abs(total_sum)

    # Find appropriate range for root finder
    npoints = 100
    a_grid = np.linspace(0.0, a_total, num=npoints)
    i = 0  # initial value
    while True:
        if i >= npoints:
            raise Exception('Root not found within range.')
        a = a_grid[i]
        curval = _fraction_of_total_function_ellip(
            a, image, center, elongation, theta, fraction, total_sum)
        if curval == 0:
            warnings.warn('Found root by pure chance!', AstropyUserWarning)
            return r, flag
        elif curval < 0:
            a_min = a
        elif curval > 0:
            a_max = a
            break
        i += 1

    a = opt.brentq(_fraction_of_total_function_ellip, a_min, a_max,
                   args=(image, center, elongation, theta, fraction, total_sum),
                   xtol=1e-6)

    return a, flag

class ConvolvedSersic2D(models.Sersic2D):
    """
    Two-dimensional Sersic surface brightness profile, convolved with
    a PSF provided by the user as a numpy array.

    See Also
    --------
    astropy.modeling.models.Sersic2D

    """
    psf = None

    @classmethod
    def set_psf(cls, psf):
        """
        Specify the PSF to be convolved with the Sersic2D model.
        """
        cls.psf = psf / np.sum(psf)  # make sure it's normalized

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """
        Evaluate the ConvolvedSersic2D model.
        """
        z_sersic = models.Sersic2D.evaluate(x, y, amplitude, r_eff, n, x_0, y_0,
                                            ellip, theta)
        if cls.psf is None:
            raise Exception('Must specify PSF using set_psf method.')

        return scipy.signal.fftconvolve(z_sersic, cls.psf, mode='same')


class SourceMorphology(object):
    """
    Class to measure the morphological parameters of a single labeled
    source. The parameters can be accessed as attributes or keys.

    Parameters
    ----------
    image : array-like
        A 2D image containing the sources of interest.
        The image must already be background-subtracted.
    segmap : array-like (int) or ``photutils.SegmentationImage``
        A 2D segmentation map where different sources are 
        labeled with different positive integer values.
        A value of zero is reserved for the background.
        It is assumed that the sum of pixel values within each
        labeled segment is positive.
    label : int
        A label indicating the source of interest.
    mask : array-like (bool), optional
        A 2D array with the same size as ``image``, where pixels
        set to ``True`` are ignored from all calculations.
    weightmap : array-like, optional
        Also known as the "sigma" image, this is a 2D array with the
        same size and units as ``image`` that contains one standard
        deviation of the value at each pixel (which is related to the
        Poisson noise). Note that SExtractor and other software
        sometimes produce a weight map in units of the variance (RMS^2)
        or the inverse variance (1/RMS^2).
    gain : scalar, optional
        A multiplication factor that converts the image units into
        electrons/pixel, which is used internally to calculate the
        weight map (i.e., the sigma-image) using Poisson statistics.
        This parameter is only used when ``weightmap`` is not provided.
    psf : array-like, optional
        A 2D array representing the PSF, where the central pixel
        corresponds to the center of the PSF. Typically, including
        this keyword argument will make the code run slower by a
        factor of a few, depending on the size of the PSF, but the
        resulting Sersic fits will in principle be more correct.
    cutout_extent : float, optional
        The target fractional size of the data cutout relative to
        the minimal bounding box containing the source. The value
        must be >= 1.
    min_cutout_size : int, optional
        The minimum size of the cutout, in case ``cutout_extent`` times
        the size of the minimal bounding box is smaller than this.
        Any given value will be truncated to an even number.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a pixel as an
        outlier, relative to its 8 neighbors. Outlying pixels are
        removed as described in Lotz et al. (2004) If the value is zero
        or negative, outliers are not removed. The default value is 10.
    annulus_width : float, optional
        The width (in pixels) of the annuli used to calculate the
        Petrosian radius and other quantities. The default value is 1.0.
    eta : float, optional
        The Petrosian ``eta`` parameter used to define the Petrosian
        radius. For a circular or elliptical aperture at the Petrosian
        radius, the mean flux at the edge of the aperture divided by
        the mean flux within the aperture is equal to ``eta``. The
        default value is typically set to 0.2 (Petrosian 1976).
    petro_fraction_gini : float, optional
        In the Gini calculation, this is the fraction of the Petrosian
        "radius" used as a smoothing scale in order to define the pixels
        that belong to the galaxy. The default value is 0.2.
    skybox_size : int, optional
        The target size (in pixels) of the "skybox" used to characterize
        the sky background.
    petro_extent_cas : float, optional
        The radius of the circular aperture used for the asymmetry
        calculation, in units of the circular Petrosian radius. The
        default value is 1.5.
    petro_fraction_cas : float, optional
        In the CAS calculations, this is the fraction of the Petrosian
        radius used as a smoothing scale. The default value is 0.25.
    boxcar_size_mid : float, optional
        In the MID calculations, this is the size (in pixels)
        of the constant kernel used to regularize the MID segmap.
        The default value is 3.0.
    niter_bh_mid : int, optional
        When calculating the multimode statistic, this is the number of
        iterations in the basin-hopping stage of the maximization.
        A value of at least 100 is recommended for "production" runs,
        but it can be time-consuming. The default value is 5.
    sigma_mid : float, optional
        In the MID calculations, this is the smoothing scale (in pixels)
        used to compute the intensity (I) statistic. The default is 1.0.
    petro_extent_flux : float, optional
        The number of Petrosian radii used to define the aperture over
        which the flux is measured. This is also used to define the inner
        "radius" of the elliptical aperture used to estimate the sky
        background in the shape asymmetry calculation. Following SDSS,
        the default value is 2.0.
    boxcar_size_shape_asym : float, optional
        When calculating the shape asymmetry segmap, this is the size
        (in pixels) of the constant kernel used to regularize the segmap.
        The default value is 3.0.
    sersic_maxiter : int, optional
        The maximum number of iterations during the Sersic profile
        fitting.
    segmap_overlap_ratio : float, optional
        The minimum ratio between the area of the intersection of
        all 3 segmaps and the area of the largest segmap in order to
        have a good measurement.

    References
    ----------
    See `README.rst` for a list of references.

    """
    def __init__(self, image, segmap, label, mask=None, weightmap=None,
                 gain=None, psf=None, cutout_extent=1.5, min_cutout_size=48,
                 n_sigma_outlier=10, annulus_width=1.0,
                 eta=0.2, petro_fraction_gini=0.2, skybox_size=32,
                 petro_extent_cas=1.5, petro_fraction_cas=0.25,
                 boxcar_size_mid=3.0, niter_bh_mid=5, sigma_mid=1.0,
                 petro_extent_flux=2.0, boxcar_size_shape_asym=3.0,
                 sersic_maxiter=500, segmap_overlap_ratio=0.5):
        self._image = image
        self._segmap = segmap
        self.label = label
        self._mask = mask
        self._weightmap = weightmap
        self._gain = gain
        self._psf = psf
        self._cutout_extent = cutout_extent
        self._min_cutout_size = min_cutout_size - min_cutout_size % 2
        self._n_sigma_outlier = n_sigma_outlier
        self._annulus_width = annulus_width
        self._eta = eta
        self._petro_fraction_gini = petro_fraction_gini
        self._skybox_size = skybox_size
        self._petro_extent_cas = petro_extent_cas
        self._petro_fraction_cas = petro_fraction_cas
        self._boxcar_size_mid = boxcar_size_mid
        self._niter_bh_mid = niter_bh_mid
        self._sigma_mid = sigma_mid
        self._petro_extent_flux = petro_extent_flux
        self._boxcar_size_shape_asym = boxcar_size_shape_asym
        self._sersic_maxiter = sersic_maxiter
        self._segmap_overlap_ratio = segmap_overlap_ratio

        # Measure runtime
        start = time.time()

        if not isinstance(self._segmap, photutils.SegmentationImage):
            self._segmap = photutils.SegmentationImage(self._segmap)

        # Check sanity of input data
        if float(photutils.__version__) < 0.5:
            self._segmap.check_label(self.label)
        else:
            self._segmap.check_labels([self.label])
        assert self._segmap.data.shape == self._image.shape
        if self._mask is not None:
            assert self._mask.shape == self._image.shape
            assert self._mask.dtype == np.bool8
        if self._weightmap is not None:
            assert self._weightmap.shape == self._image.shape

        # Normalize PSF
        if self._psf is not None:
            self._psf = self._psf / np.sum(self._psf)

        # Check that the labeled galaxy segment has a positive flux sum:
        assert np.sum(self._cutout_stamp_maskzeroed_no_bg) > 0

        # These flags will be modified during the calculations:
        self.flag = 0  # attempts to flag bad measurements
        self.flag_sersic = 0  # attempts to flag bad Sersic fits

        # If something goes wrong, use centroid instead of asymmetry center
        # (better performance in some pathological cases, e.g. GOODS-S 32143):
        self._use_centroid = False

        # Centroid of the source relative to the "postage stamp" cutout:
        self._xc_stamp = self.xc_centroid - self.xmin_stamp
        self._yc_stamp = self.yc_centroid - self.ymin_stamp

        # Print warning if centroid is masked:
        ic, jc = int(self._yc_stamp), int(self._xc_stamp)
        if self._cutout_stamp_maskzeroed[ic, jc] == 0:
            warnings.warn('Centroid is masked.', AstropyUserWarning)
            self.flag = 1

        # Position of the source's brightest pixel relative to the stamp cutout:
        maxval = np.max(self._cutout_stamp_maskzeroed_no_bg)
        maxval_stamp_pos = np.argwhere(self._cutout_stamp_maskzeroed_no_bg == maxval)[0]
        self._x_maxval_stamp = maxval_stamp_pos[1]
        self._y_maxval_stamp = maxval_stamp_pos[0]

        # ------------------------------------------------------------------
        # NOTE: no morphology calculations have been done so far, but
        # this will change below this line. Modify this __init__ with care.
        # ------------------------------------------------------------------

        # For simplicity, evaluate all "lazy" properties at once:
        self._calculate_morphology()

        # Check if image is background-subtracted; set flag=1 if not.
        if np.abs(self.sky_mean) > self.sky_sigma:
            warnings.warn('Image is not background-subtracted.', AstropyUserWarning)
            self.flag = 1

        # Check segmaps and set flag=1 if they are very different
        self._check_segmaps()

        # Save runtime
        self.runtime = time.time() - start

    def __getitem__(self, key):
        return getattr(self, key)

    def _get_badpixels(self, image):
        """
        Detect outliers (bad pixels) as described in Lotz et al. (2004).

        Notes
        -----
        ndi.generic_filter(image, np.std, ...) is too slow,
        so we do a workaround using ndi.convolve.
        """
        # Pixel weights, excluding central pixel.
        w = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]], dtype=np.float64)
        w = w / np.sum(w)
        
        # Use the fact that var(x) = <x^2> - <x>^2.
        local_mean = ndi.convolve(image, w)
        local_mean2 = ndi.convolve(image**2, w)
        local_std = np.sqrt(local_mean2 - local_mean**2)

        # Get "bad pixels"
        badpixels = (np.abs(image - local_mean) >
                      self._n_sigma_outlier * local_std)
        
        return badpixels

    def _calculate_morphology(self):
        """
        Calculate all morphological parameters, which are stored
        as "lazy" properties.
        """
        quantities = [
            'xc_centroid',
            'yc_centroid',
            'ellipticity_centroid',
            'elongation_centroid',
            'orientation_centroid',
            'xc_asymmetry',
            'yc_asymmetry',
            'ellipticity_asymmetry',
            'elongation_asymmetry',
            'orientation_asymmetry',
            'flux_circ',
            'flux_ellip',
            'rpetro_circ',
            'rpetro_ellip',
            'rmax_circ',
            'rmax_ellip',
            'rhalf_circ',
            'rhalf_ellip',
            'r20',
            'r50',
            'r80',
            'gini',
            'm20',
            'gini_m20_bulge',
            'gini_m20_merger',
            'sn_per_pixel',
            'concentration',
            'asymmetry',
            'smoothness',
            'multimode',
            'intensity',
            'deviation',
            'outer_asymmetry',
            'shape_asymmetry',
            'sersic_amplitude',
            'sersic_rhalf',
            'sersic_n',
            'sersic_xc',
            'sersic_yc',
            'sersic_ellip',
            'sersic_theta',
            'sky_mean',
            'sky_median',
            'sky_sigma',
            'xmin_stamp',
            'ymin_stamp',
            'xmax_stamp',
            'ymax_stamp',
            'nx_stamp',
            'ny_stamp',
        ]
        for q in quantities:
            tmp = self[q]

    def _check_segmaps(self):
        """
        Compare Gini segmap and MID segmap; set flag=1 if they are
        very different from each other.
        """
        area_max = max(np.sum(self._segmap_gini),
                       np.sum(self._segmap_mid))
        area_overlap = np.sum(self._segmap_gini &
                              self._segmap_mid)
        if area_max == 0:
            warnings.warn('Segmaps are empty!', AstropyUserWarning)
            self.flag = 1
            return

        area_ratio = area_overlap / float(area_max)
        if area_ratio < self._segmap_overlap_ratio:
            warnings.warn('Gini and MID segmaps are quite different.',
                          AstropyUserWarning)
            self.flag = 1

    @lazyproperty
    def _centroid(self):
        """
        The (yc, xc) centroid of the input segment, relative to
        ``_slice_stamp``.
        """
        image = np.float64(self._cutout_stamp_maskzeroed_no_bg)  # skimage wants double

        # Calculate centroid
        M = skimage.measure.moments(image, order=1)
        assert M[0, 0] > 0  # already checked by constructor
        yc = M[1, 0] / M[0, 0]
        xc = M[0, 1] / M[0, 0]
        yc += 0.5; xc += 0.5  # shift pixel positions

        return np.array([xc, yc])

    @lazyproperty
    def xc_centroid(self):
        """
        The x-coordinate of the centroid, relative to the original image.
        """
        return self._centroid[0] + self.xmin_stamp

    @lazyproperty
    def yc_centroid(self):
        """
        The y-coordinate of the centroid, relative to the original image.
        """
        return self._centroid[1] + self.ymin_stamp

    def _covariance_generic(self, xc, yc):
        """
        The covariance matrix of a Gaussian function that has the same
        second-order moments as the source, with respect to ``(xc, yc)``.
        """
        # skimage wants double precision:
        image = np.float64(self._cutout_stamp_maskzeroed_no_bg)

        # Calculate moments w.r.t. given center
        xc = xc - self.xmin_stamp - 0.5  # w.r.t. lower-left corner of pixels
        yc = yc - self.ymin_stamp - 0.5
        Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
        assert Mc[0, 0] > 0

        covariance = np.array([
            [Mc[0, 2], Mc[1, 1]],
            [Mc[1, 1], Mc[2, 0]]])
        covariance /= Mc[0, 0]  # normalize

        # If there are nonpositive second moments, we deal with them,
        # but we indicate that there's something wrong with the data.
        if (covariance[0, 0] <= 0) or (covariance[1, 1] <= 0):
            warnings.warn('Nonpositive second moment.', AstropyUserWarning)
            self.flag = 1

        # Modify covariance matrix in case of "infinitely thin" sources
        # by iteratively increasing the diagonal elements (see SExtractor
        # manual, eq. 43). Note that we allow negative moments.
        rho = 1.0 / 12.0  # variance of 1 pixel-wide top-hat distribution
        x2, xy, xy, y2 = covariance.flat
        while np.abs(x2*y2 - xy**2) < rho**2:
            x2 += (x2 >= 0) * rho - (x2 < 0) * rho  # np.sign(0) == 0 is no good
            y2 += (y2 >= 0) * rho - (y2 < 0) * rho
        covariance = np.array([[x2, xy],
                               [xy, y2]])

        return covariance

    def _eigvals_generic(self, covariance):
        """
        The ordered (largest first) eigenvalues of the covariance
        matrix, which correspond to the *squared* semimajor and
        semiminor axes. Note that we allow negative eigenvalues.
        """
        eigvals = np.linalg.eigvals(covariance)
        eigvals = np.sort(np.abs(eigvals))[::-1]  # largest first (by abs. value)

        # We deal with negative eigenvalues, but we indicate that something
        # is not OK with the data (eigenvalues cannot be exactly zero after
        # the SExtractor-like regularization routine).
        if np.any(eigvals < 0):
            warnings.warn('Some negative eigenvalues.', AstropyUserWarning)
            self.flag = 1

        return eigvals

    def _ellipticity_generic(self, eigvals):
        """
        The ellipticity of (the Gaussian function that has the same
        second-order moments as) the source. Note that we allow
        negative eigenvalues.
        """
        a = np.sqrt(np.abs(eigvals[0]))
        b = np.sqrt(np.abs(eigvals[1]))

        return 1.0 - (b / a)

    def _elongation_generic(self, eigvals):
        """
        The elongation of (the Gaussian function that has the same
        second-order moments as) the source. Note that we allow
        negative eigenvalues.
        """
        a = np.sqrt(np.abs(eigvals[0]))
        b = np.sqrt(np.abs(eigvals[1]))

        return a / b

    def _orientation_generic(self, covariance):
        """
        The orientation (in radians) of the source.
        """
        x2, xy, xy, y2 = covariance.flat
        
        # SExtractor manual, eq. (21):
        theta = 0.5 * np.arctan2(2.0 * xy, x2 - y2)
        
        return theta

    @lazyproperty
    def _covariance_centroid(self):
        """
        The covariance matrix of a Gaussian function that has the same
        second-order moments as the source, with respect to the centroid.
        """
        return self._covariance_generic(self.xc_centroid, self.yc_centroid)

    @lazyproperty
    def _eigvals_centroid(self):
        """
        The ordered (largest first) eigenvalues of the covariance
        matrix, which correspond to the *squared* semimajor and
        semiminor axes, relative to the centroid.
        """
        return self._eigvals_generic(self._covariance_centroid)

    @lazyproperty
    def ellipticity_centroid(self):
        """
        The ellipticity of (the Gaussian function that has the same
        second-order moments as) the source, relative to the centroid.
        """
        return self._ellipticity_generic(self._eigvals_centroid)

    @lazyproperty
    def elongation_centroid(self):
        """
        The elongation of (the Gaussian function that has the same
        second-order moments as) the source, relative to the centroid.
        """
        return self._elongation_generic(self._eigvals_centroid)

    @lazyproperty
    def orientation_centroid(self):
        """
        The orientation (in radians) of the source, relative to the
        centroid.
        """
        return self._orientation_generic(self._covariance_centroid)

    @lazyproperty
    def _slice_stamp(self):
        """
        Attempt to create a square slice that is somewhat larger
        than the minimal bounding box containing the labeled segment.
        Note that the cutout may not be square when the source is
        close to a border of the original image.
        """
        assert self._cutout_extent >= 1.0
        assert self._min_cutout_size >= 2
        # Get dimensions of original bounding box
        s = self._segmap.slices[self.label - 1]
        xmin, xmax = s[1].start, s[1].stop - 1
        ymin, ymax = s[0].start, s[0].stop - 1
        dx, dy = xmax + 1 - xmin, ymax + 1 - ymin
        xc, yc = xmin + dx//2, ymin + dy//2
        # Add some extra space in each dimension
        dist = int(max(dx, dy) * self._cutout_extent / 2.0)
        # Make sure that cutout size is at least ``min_cutout_size``
        dist = max(dist, self._min_cutout_size // 2)
        # Make cutout
        ny, nx = self._image.shape
        slice_stamp = (slice(max(0, yc-dist), min(ny, yc+dist)),
                       slice(max(0, xc-dist), min(nx, xc+dist)))
        return slice_stamp

    @lazyproperty
    def xmin_stamp(self):
        """
        The minimum ``x`` position of the 'postage stamp'.
        """
        return self._slice_stamp[1].start

    @lazyproperty
    def ymin_stamp(self):
        """
        The minimum ``y`` position of the 'postage stamp'.
        """
        return self._slice_stamp[0].start

    @lazyproperty
    def xmax_stamp(self):
        """
        The maximum ``x`` position of the 'postage stamp'.
        """
        return self._slice_stamp[1].stop - 1

    @lazyproperty
    def ymax_stamp(self):
        """
        The maximum ``y`` position of the 'postage stamp'.
        """
        return self._slice_stamp[0].stop - 1

    @lazyproperty
    def nx_stamp(self):
        """
        Number of pixels in the 'postage stamp' along the ``x`` direction.
        """
        return self.xmax_stamp + 1 - self.xmin_stamp

    @lazyproperty
    def ny_stamp(self):
        """
        Number of pixels in the 'postage stamp' along the ``y`` direction.
        """
        return self.ymax_stamp + 1 - self.ymin_stamp

    @lazyproperty
    def _mask_stamp_nan(self):
        """
        Flag any NaN or inf values within the postage stamp.
        """
        locs_invalid = ~np.isfinite(self._image[self._slice_stamp])
        if self._weightmap is not None:
            locs_invalid |= ~np.isfinite(self._weightmap[self._slice_stamp])
        return locs_invalid

    @lazyproperty
    def _mask_stamp_badpixels(self):
        """
        Flag badpixels (outliers).
        """
        self.num_badpixels = -1
        badpixels = np.zeros((self.ny_stamp, self.nx_stamp), dtype=np.bool8)
        if self._n_sigma_outlier > 0:
            badpixels = self._get_badpixels(self._image[self._slice_stamp])
            self.num_badpixels = np.sum(badpixels)
        return badpixels

    @lazyproperty
    def _mask_stamp(self):
        """
        Create a total binary mask for the "postage stamp".
        Pixels belonging to other sources (as well as pixels masked
        using the ``mask`` keyword argument) are set to ``True``,
        but the background (segmap == 0) is left alone.
        """
        segmap_stamp = self._segmap.data[self._slice_stamp]
        mask_stamp = (segmap_stamp != 0) & (segmap_stamp != self.label)
        if self._mask is not None:
            mask_stamp |= self._mask[self._slice_stamp]
        mask_stamp |= self._mask_stamp_nan
        mask_stamp |= self._mask_stamp_badpixels
        return mask_stamp

    @lazyproperty
    def _mask_stamp_no_bg(self):
        """
        Similar to ``_mask_stamp``, but also mask the background.
        """
        segmap_stamp = self._segmap.data[self._slice_stamp]
        return self._mask_stamp | (segmap_stamp == 0)

    @lazyproperty
    def _cutout_stamp_maskzeroed(self):
        """
        Return a data cutout with its shape and position determined
        by ``_slice_stamp``. Pixels belonging to other sources
        (as well as pixels where ``mask`` == 1) are set to zero,
        but the background is left alone.
        
        In addition, NaN or inf values are removed at this point,
        as well as badpixels (outliers).
        """
        return np.where(~self._mask_stamp,
                        self._image[self._slice_stamp], 0.0)

    @lazyproperty
    def _cutout_stamp_maskzeroed_no_bg(self):
        """
        Like ``_cutout_stamp_maskzeroed``, but also mask the
        background.
        """
        return np.where(~self._mask_stamp_no_bg,
                        self._image[self._slice_stamp], 0.0)

    @lazyproperty
    def _weightmap_stamp(self):
        """
        Return a cutout of the weight map over the "postage stamp" region.
        If a weightmap is not provided as input, it is created using the
        ``gain`` argument.
        """
        if self._weightmap is None:
            if self._gain is None:
                raise Exception('Must provide either weightmap or gain.')
            else:
                assert self._gain > 0
                weightmap_stamp = np.sqrt(
                    np.abs(self._image[self._slice_stamp])/self._gain + self.sky_sigma**2)
        else:
            weightmap_stamp = self._weightmap[self._slice_stamp]

        weightmap_stamp[self._mask_stamp_nan] = 0.0
        return weightmap_stamp

    @lazyproperty
    def _sorted_pixelvals_stamp_no_bg(self):
        """
        The sorted pixel values of the (zero-masked) postage stamp,
        excluding masked values and the background.
        """
        return np.sort(self._cutout_stamp_maskzeroed[~self._mask_stamp_no_bg])

    @lazyproperty
    def _cutout_stamp_maskzeroed_no_bg_nonnegative(self):
        """
        Same as ``_cutout_stamp_maskzeroed_no_bg``, but masking
        negative pixels.
        """
        image = self._cutout_stamp_maskzeroed_no_bg
        return np.where(image > 0, image, 0.0)

    @lazyproperty
    def _sorted_pixelvals_stamp_no_bg_nonnegative(self):
        """
        Same as ``_sorted_pixelvals_stamp_no_bg``, but masking
        negative pixels.
        """
        image = self._cutout_stamp_maskzeroed_no_bg_nonnegative
        return np.sort(image[~self._mask_stamp_no_bg])

    @lazyproperty
    def _diagonal_distance(self):
        """
        Return the diagonal distance (in pixels) of the postage stamp.
        This is used as an upper bound in some calculations.
        """
        ny, nx = self._cutout_stamp_maskzeroed.shape
        return np.sqrt(nx**2 + ny**2)

    def _petrosian_function_circ(self, r, center):
        """
        Helper function to calculate the circular Petrosian radius.
        
        For a given radius ``r``, return the ratio of the mean flux
        over a circular annulus divided by the mean flux within the
        circle, minus "eta" (eq. 4 from Lotz et al. 2004). The root
        of this function is the Petrosian radius.
        """
        image = self._cutout_stamp_maskzeroed

        r_in = r - 0.5 * self._annulus_width
        r_out = r + 0.5 * self._annulus_width

        circ_annulus = photutils.CircularAnnulus(center, r_in, r_out)
        circ_aperture = photutils.CircularAperture(center, r)

        # Force mean fluxes to be positive:
        circ_annulus_mean_flux = np.abs(_aperture_mean_nomask(
            circ_annulus, image, method='exact'))
        circ_aperture_mean_flux = np.abs(_aperture_mean_nomask(
            circ_aperture, image, method='exact'))
        
        if circ_aperture_mean_flux == 0:
            warnings.warn('[rpetro_circ] Mean flux is zero.', AstropyUserWarning)
            ratio = 1.0
            self.flag = 1
        else:
            ratio = circ_annulus_mean_flux / circ_aperture_mean_flux

        return ratio - self._eta

    def _rpetro_circ_generic(self, center):
        """
        Compute the Petrosian radius for concentric circular apertures.

        Notes
        -----
        The so-called "curve of growth" is not always monotonic,
        e.g., when there is a bright, unlabeled and unmasked
        secondary source in the image, so we cannot just apply a
        root-finding algorithm over the full interval.
        Instead, we proceed in two stages: first we do a coarse,
        brute-force search for an appropriate interval (that
        contains a root), and then we apply the root-finder.

        """
        # Find appropriate range for root finder
        npoints = 100
        r_inner = self._annulus_width
        r_outer = self._diagonal_distance
        assert r_inner < r_outer
        dr = (r_outer - r_inner) / float(npoints-1)
        r_min, r_max = None, None
        r = r_inner  # initial value
        while True:
            if r >= r_outer:
                warnings.warn('[rpetro_circ] rpetro larger than cutout.',
                              AstropyUserWarning)
                self.flag = 1
            curval = self._petrosian_function_circ(r, center)
            if curval == 0:
                warnings.warn('[rpetro_circ] We found rpetro by pure chance!',
                              AstropyUserWarning)
                return r
            elif curval > 0:
                r_min = r
            elif curval < 0:
                if r_min is None:
                    warnings.warn('[rpetro_circ] r_min is not defined yet.',
                                  AstropyUserWarning)
                    self.flag = 1
                else:
                    r_max = r
                    break
            r += dr

        rpetro_circ = opt.brentq(self._petrosian_function_circ, 
                                 r_min, r_max, args=(center,), xtol=1e-6)

        return rpetro_circ

    @lazyproperty
    def _rpetro_circ_centroid(self):
        """
        Calculate the Petrosian radius with respect to the centroid.
        This is only used as a preliminary value for the asymmetry
        calculation.
        """
        center = np.array([self._xc_stamp, self._yc_stamp])
        return self._rpetro_circ_generic(center)

    @lazyproperty
    def rpetro_circ(self):
        """
        Calculate the Petrosian radius with respect to the point
        that minimizes the asymmetry.
        """
        return self._rpetro_circ_generic(self._asymmetry_center)

    @lazyproperty
    def flux_circ(self):
        """
        Return the sum of the pixel values over a circular aperture
        with radius equal to ``petro_extent_flux`` (usually 2) times
        the circular Petrosian radius.
        """
        image = self._cutout_stamp_maskzeroed
        r = self._petro_extent_flux * self.rpetro_circ
        ap = photutils.CircularAperture(self._asymmetry_center, r)
        # Force flux sum to be positive:
        ap_sum = np.abs(ap.do_photometry(image, method='exact')[0][0])
        return ap_sum

    def _petrosian_function_ellip(self, a, center, elongation, theta):
        """
        Helper function to calculate the Petrosian "radius".
        
        For the ellipse with semi-major axis ``a``, return the
        ratio of the mean flux over an elliptical annulus
        divided by the mean flux within the ellipse,
        minus "eta" (eq. 4 from Lotz et al. 2004). The root of
        this function is the Petrosian "radius".
        
        """
        image = self._cutout_stamp_maskzeroed

        b = a / elongation
        a_in = a - 0.5 * self._annulus_width
        a_out = a + 0.5 * self._annulus_width

        b_out = a_out / elongation

        ellip_annulus = photutils.EllipticalAnnulus(
            center, a_in, a_out, b_out, theta)
        ellip_aperture = photutils.EllipticalAperture(
            center, a, b, theta)

        # Force mean fluxes to be positive:
        ellip_annulus_mean_flux = np.abs(_aperture_mean_nomask(
            ellip_annulus, image, method='exact'))
        ellip_aperture_mean_flux = np.abs(_aperture_mean_nomask(
            ellip_aperture, image, method='exact'))

        if ellip_aperture_mean_flux == 0:
            warnings.warn('[rpetro_ellip] Mean flux is zero.', AstropyUserWarning)
            ratio = 1.0
            self.flag = 1
        else:
            ratio = ellip_annulus_mean_flux / ellip_aperture_mean_flux

        return ratio - self._eta

    def _rpetro_ellip_generic(self, center, elongation, theta):
        """
        Compute the Petrosian "radius" (actually the semi-major axis)
        for concentric elliptical apertures.
        
        Notes
        -----
        The so-called "curve of growth" is not always monotonic,
        e.g., when there is a bright, unlabeled and unmasked
        secondary source in the image, so we cannot just apply a
        root-finding algorithm over the full interval.
        Instead, we proceed in two stages: first we do a coarse,
        brute-force search for an appropriate interval (that
        contains a root), and then we apply the root-finder.

        """
        # Find appropriate range for root finder
        npoints = 100
        a_inner = self._annulus_width
        a_outer = self._diagonal_distance
        assert a_inner < a_outer
        da = (a_outer - a_inner) / float(npoints-1)
        a_min, a_max = None, None
        a = a_inner  # initial value
        while True:
            if a >= a_outer:
                warnings.warn('[rpetro_ellip] rpetro larger than cutout.',
                              AstropyUserWarning)
                self.flag = 1
            curval = self._petrosian_function_ellip(a, center, elongation, theta)
            if curval == 0:
                warnings.warn('[rpetro_ellip] We found rpetro by pure chance!',
                              AstropyUserWarning)
                return a
            elif curval > 0:
                a_min = a
            elif curval < 0:
                if a_min is None:
                    warnings.warn('[rpetro_ellip] a_min is not defined yet.',
                                  AstropyUserWarning)
                    self.flag = 1
                else:
                    a_max = a
                    break
            a += da

        rpetro_ellip = opt.brentq(self._petrosian_function_ellip, a_min, a_max,
                                  args=(center, elongation, theta,), xtol=1e-6)

        return rpetro_ellip

    @lazyproperty
    def rpetro_ellip(self):
        """
        Return the elliptical Petrosian "radius", calculated with
        respect to the point that minimizes the asymmetry.
        """
        return self._rpetro_ellip_generic(
            self._asymmetry_center, self.elongation_asymmetry,
            self.orientation_asymmetry)

    @lazyproperty
    def flux_ellip(self):
        """
        Return the sum of the pixel values over an elliptical aperture
        with "radius" equal to ``petro_extent_flux`` (usually 2) times
        the elliptical Petrosian "radius".
        """
        image = self._cutout_stamp_maskzeroed
        a = self._petro_extent_flux * self.rpetro_ellip
        b = a / self.elongation_asymmetry
        theta = self.orientation_asymmetry
        ap = photutils.EllipticalAperture(self._asymmetry_center, a, b, theta)
        # Force flux sum to be positive:
        ap_sum = np.abs(ap.do_photometry(image, method='exact')[0][0])
        return ap_sum

    #######################
    # Gini-M20 statistics #
    #######################

    @lazyproperty
    def _segmap_gini(self):
        """
        Create a new segmentation map (relative to the "postage stamp")
        based on the elliptical Petrosian radius.
        """
        # Smooth image
        petro_sigma = self._petro_fraction_gini * self.rpetro_ellip
        cutout_smooth = ndi.gaussian_filter(self._cutout_stamp_maskzeroed, petro_sigma)

        # Use mean flux at the Petrosian "radius" as threshold
        a_in = self.rpetro_ellip - 0.5 * self._annulus_width
        a_out = self.rpetro_ellip + 0.5 * self._annulus_width
        b_out = a_out / self.elongation_asymmetry
        theta = self.orientation_asymmetry
        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_stamp, self._yc_stamp), a_in, a_out, b_out, theta)
        ellip_annulus_mean_flux = _aperture_mean_nomask(
            ellip_annulus, cutout_smooth, method='exact')

        above_threshold = cutout_smooth >= ellip_annulus_mean_flux

        # Grow regions with 8-connected neighbor "footprint"
        s = ndi.generate_binary_structure(2, 2)
        labeled_array, num_features = ndi.label(above_threshold, structure=s)
        
        # In some rare cases (e.g., Pan-STARRS J020218.5+672123_g.fits.gz),
        # this results in an empty segmap, so there is nothing to do.
        if num_features == 0:
            warnings.warn('[segmap_gini] Empty Gini segmap!',
                          AstropyUserWarning)
            self.flag = 1
            return above_threshold

        # In other cases (e.g., object 110 from CANDELS/GOODS-S WFC/F160W),
        # the Gini segmap occupies the entire image, which is also not OK.
        if np.sum(above_threshold) == cutout_smooth.size:
            warnings.warn('[segmap_gini] Full Gini segmap!',
                          AstropyUserWarning)
            self.flag = 1
            return above_threshold

        # If more than one region, activate the "bad measurement" flag
        # and only keep segment that contains the brightest pixel.
        if num_features > 1:
            warnings.warn('[segmap_gini] Disjoint features in Gini segmap.',
                          AstropyUserWarning)
            self.flag = 1
            ic, jc = np.argwhere(cutout_smooth == np.max(cutout_smooth))[0]
            assert labeled_array[ic, jc] != 0
            segmap = labeled_array == labeled_array[ic, jc]
        else:
            segmap = above_threshold

        return segmap

    @lazyproperty
    def gini(self):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        """
        image = self._cutout_stamp_maskzeroed.flatten()
        segmap = self._segmap_gini.flatten()

        sorted_pixelvals = np.sort(np.abs(image[segmap]))
        n = len(sorted_pixelvals)
        if n <= 1 or np.sum(sorted_pixelvals) == 0:
            warnings.warn('[gini] Not enough data for Gini calculation.',
                          AstropyUserWarning)

            self.flag = 1
            return -99.0  # invalid
        
        indices = np.arange(1, n+1)  # start at i=1
        gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
                (float(n-1) * np.sum(sorted_pixelvals)))

        return gini

    @lazyproperty
    def m20(self):
        """
        Calculate the M_20 coefficient as described in Lotz et al. (2004).
        """
        if np.sum(self._segmap_gini) == 0:
            return -99.0  # invalid

        # Use the same region as in the Gini calculation
        image = np.where(self._segmap_gini, self._cutout_stamp_maskzeroed, 0.0)
        image = np.float64(image)  # skimage wants double

        # Calculate centroid
        M = skimage.measure.moments(image, order=1)
        if M[0, 0] <= 0:
            warnings.warn('[deviation] Nonpositive flux within Gini segmap.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid
        yc = M[1, 0] / M[0, 0]
        xc = M[0, 1] / M[0, 0]
        # Note that we do not shift (yc, xc) by 0.5 pixels here, since
        # (yc, xc) is only used as input for other skimage functions.

        # Calculate second total central moment
        Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
        second_moment_tot = Mc[0, 2] + Mc[2, 0]

        # Calculate threshold pixel value
        sorted_pixelvals = np.sort(image.flatten())
        flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
        sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
        if len(sorted_pixelvals_20) == 0:
            # This can happen when there are very few pixels.
            warnings.warn('[m20] Not enough data for M20 calculation.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid
        threshold = sorted_pixelvals_20[0]

        # Calculate second moment of the brightest pixels
        image_20 = np.where(image >= threshold, image, 0.0)
        Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
        second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

        if (second_moment_20 <= 0) | (second_moment_tot <= 0):
            warnings.warn('[m20] Negative second moment(s).',
                          AstropyUserWarning)
            self.flag = 1
            m20 = -99.0  # invalid
        else:
            m20 = np.log10(second_moment_20 / second_moment_tot)

        return m20

    @lazyproperty
    def gini_m20_bulge(self):
        """
        Return the Gini-M20 bulge statistic, F(G, M20), as defined
        in Rodriguez-Gomez et al. (2018).
        """
        if (self.gini == -99.0) or (self.m20 == -99.0):
            return -99.0  # invalid

        return -0.693*self.m20 + 4.95*self.gini - 3.96

    @lazyproperty
    def gini_m20_merger(self):
        """
        Return the Gini-M20 merger statistic, S(G, M20), as defined
        in Rodriguez-Gomez et al. (2018).
        """
        if (self.gini == -99.0) or (self.m20 == -99.0):
            return -99.0  # invalid

        return 0.139*self.m20 + 0.990*self.gini - 0.327

    @lazyproperty
    def sn_per_pixel(self):
        """
        Calculate the signal-to-noise per pixel using the Petrosian segmap.
        """
        weightmap = self._weightmap_stamp
        if np.any(weightmap < 0):
            warnings.warn('[sn_per_pixel] Some negative weightmap values.',
                          AstropyUserWarning)
            weightmap = np.abs(weightmap)

        locs = self._segmap_gini & (self._cutout_stamp_maskzeroed >= 0)
        pixelvals = self._cutout_stamp_maskzeroed[locs]
        # The sky background noise is already included in the weightmap:
        snp = np.mean(pixelvals / weightmap[locs])

        if not np.isfinite(snp):
            warnings.warn('Invalid sn_per_pixel.', AstropyUserWarning)
            self.flag = 1
            snp = -99.0  # invalid

        return snp

    ##################
    # CAS statistics #
    ##################

    @lazyproperty
    def _slice_skybox(self):
        """
        Try to find a region of the sky that only contains background.
        
        In principle, a more accurate approach is possible
        (e.g. Shi et al. 2009, ApJ, 697, 1764).
        """
        segmap = self._segmap.data[self._slice_stamp]
        ny, nx = segmap.shape
        mask = np.zeros(segmap.shape, dtype=np.bool8)
        if self._mask is not None:
            mask = self._mask[self._slice_stamp]
        
        cur_skybox_size = self._skybox_size
        while True:
            for i in range(0, ny - cur_skybox_size):
                for j in range(0, nx - cur_skybox_size):
                    boxslice = (slice(i, i + cur_skybox_size),
                                slice(j, j + cur_skybox_size))
                    if np.all(segmap[boxslice] == 0) and np.all(~mask[boxslice]):
                        return boxslice

            # If we got here, a skybox of the given size was not found.
            if cur_skybox_size <= 1:
                warnings.warn('[skybox] Skybox not found.', AstropyUserWarning)
                self.flag = 1
                return (slice(0, 0), slice(0, 0))
            else:
                cur_skybox_size //= 2
                warnings.warn('[skybox] Reducing skybox size to %d.' % (
                              cur_skybox_size), AstropyUserWarning)

        # Should not reach this point.
        raise AssertionError

    @lazyproperty
    def sky_mean(self):
        """
        Mean background value. Equal to -99.0 when there is no skybox.
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        if bkg.size == 0:
            assert self.flag == 1
            return -99.0

        return np.mean(bkg)

    @lazyproperty
    def sky_median(self):
        """
        Median background value. Equal to -99.0 when there is no skybox.
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        if bkg.size == 0:
            assert self.flag == 1
            return -99.0

        return np.median(bkg)

    @lazyproperty
    def sky_sigma(self):
        """
        Standard deviation of the background. Equal to -99.0 when there
        is no skybox.
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        if bkg.size == 0:
            assert self.flag == 1
            return -99.0

        return np.std(bkg)

    @lazyproperty
    def _sky_asymmetry(self):
        """
        Asymmetry of the background. Equal to -99.0 when there is no
        skybox. Note the peculiar normalization (for reference only).
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        bkg_180 = bkg[::-1, ::-1]
        if bkg.size == 0:
            assert self.flag == 1
            return -99.0

        return np.sum(np.abs(bkg_180 - bkg)) / float(bkg.size)

    @lazyproperty
    def _sky_smoothness(self):
        """
        Smoothness of the background. Equal to -99.0 when there is no
        skybox. Note the peculiar normalization (for reference only).
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        if bkg.size == 0:
            assert self.flag == 1
            return -99.0

        # If the smoothing "boxcar" is larger than the skybox itself,
        # this just sets all values equal to the mean:
        boxcar_size = int(self._petro_fraction_cas * self.rpetro_circ)
        bkg_smooth = ndi.uniform_filter(bkg, size=boxcar_size)

        bkg_diff = bkg - bkg_smooth
        bkg_diff[bkg_diff < 0] = 0.0  # set negative pixels to zero

        return np.sum(bkg_diff) / float(bkg.size)

    def _asymmetry_function(self, center, image, kind):
        """
        Helper function to determine the asymmetry and center of asymmetry.
        The idea is to minimize the output of this function.
        
        Parameters
        ----------
        center : tuple or array-like
            The (x,y) position of the center.
        image : array-like
            The 2D image.
        kind : {'cas', 'outer', 'shape'}
            Whether to calculate the traditional CAS asymmetry (default),
            outer asymmetry or shape asymmetry.
        
        Returns
        -------
        asym : The asymmetry statistic for the given center.

        """
        image = np.float64(image)  # skimage wants double
        ny, nx = image.shape
        xc, yc = center

        if xc < 0 or xc >= nx or yc < 0 or yc >= ny:
            warnings.warn('[asym_center] Minimizer tried to exit bounds.',
                          AstropyUserWarning)
            self.flag = 1
            self._use_centroid = True
            # Return high value to keep minimizer within range:
            return 100.0

        # Rotate around given center
        image_180 = skimage.transform.rotate(image, 180.0, center=center)

        # Apply symmetric mask
        mask = self._mask_stamp.copy()
        mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
        mask_180 = mask_180 >= 0.5  # convert back to bool
        mask_symmetric = mask | mask_180
        image = np.where(~mask_symmetric, image, 0.0)
        image_180 = np.where(~mask_symmetric, image_180, 0.0)

        # Create aperture for the chosen kind of asymmetry
        if kind == 'cas':
            r = self._petro_extent_cas * self._rpetro_circ_centroid
            ap = photutils.CircularAperture(center, r)
        elif kind == 'outer':
            a_in = self.rhalf_ellip
            a_out = self.rmax_ellip
            b_out = a_out / self.elongation_asymmetry
            theta = self.orientation_asymmetry
            if np.isnan(a_in) or np.isnan(a_out) or (a_in <= 0) or (a_out <= 0):
                warnings.warn('[outer_asym] Invalid annulus dimensions.',
                              AstropyUserWarning)
                self.flag = 1
                return -99.0  # invalid
            ap = photutils.EllipticalAnnulus(center, a_in, a_out, b_out, theta)
        elif kind == 'shape':
            if np.isnan(self.rmax_circ) or (self.rmax_circ <= 0):
                warnings.warn('[shape_asym] Invalid rmax_circ value.',
                              AstropyUserWarning)
                self.flag = 1
                return -99.0  # invalid
            ap = photutils.CircularAperture(center, self.rmax_circ)
        else:
            raise Exception('Asymmetry kind not understood:', kind)

        # Apply eq. 10 from Lotz et al. (2004)
        ap_abs_sum = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(np.abs(image_180-image), method='exact')[0][0]

        if ap_abs_sum == 0.0:
            warnings.warn('[asymmetry_function] Zero flux sum.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid

        if kind == 'shape':
            # The shape asymmetry of the background is zero
            asym = ap_abs_diff / ap_abs_sum
        else:
            if self._sky_asymmetry == -99.0:  # invalid skybox
                asym = ap_abs_diff / ap_abs_sum
            else:
                ap_area = _aperture_area(ap, mask_symmetric)
                asym = (ap_abs_diff - ap_area*self._sky_asymmetry) / ap_abs_sum
                
        return asym

    @lazyproperty
    def _asymmetry_center(self):
        """
        Find the position of the central pixel (relative to the
        "postage stamp" cutout) that minimizes the (CAS) asymmetry.
        """
        center_0 = np.array([self._xc_stamp, self._yc_stamp])  # initial guess
        center_asym = opt.fmin(self._asymmetry_function, center_0,
                               args=(self._cutout_stamp_maskzeroed, 'cas'),
                               xtol=1e-6, disp=0)

        # Check if flag was activated by _asymmetry_function:
        if self._use_centroid:
            warnings.warn('Using centroid instead of asymmetry center.',
                          AstropyUserWarning)
            center_asym = center_0

        # Print warning if center is masked
        ic, jc = int(center_asym[1]), int(center_asym[0])
        if self._cutout_stamp_maskzeroed[ic, jc] == 0:
            warnings.warn('[asym_center] Asymmetry center is masked.',
                          AstropyUserWarning)
            self.flag = 1

        return center_asym

    @lazyproperty
    def xc_asymmetry(self):
        """
        The x-coordinate of the point that minimizes the asymmetry,
        relative to the original image.
        """
        return self.xmin_stamp + self._asymmetry_center[0]

    @lazyproperty
    def yc_asymmetry(self):
        """
        The y-coordinate of the point that minimizes the asymmetry,
        relative to the original image.
        """
        return self.ymin_stamp + self._asymmetry_center[1]

    @lazyproperty
    def _covariance_asymmetry(self):
        """
        The covariance matrix of a Gaussian function that has the same
        second-order moments as the source, with respect to the point
        that minimizes the asymmetry.
        """
        return self._covariance_generic(self.xc_asymmetry, self.yc_asymmetry)

    @lazyproperty
    def _eigvals_asymmetry(self):
        """
        The ordered (largest first) eigenvalues of the covariance
        matrix, which correspond to the *squared* semimajor and
        semiminor axes, relative to the point that minimizes the asymmetry.
        """
        return self._eigvals_generic(self._covariance_asymmetry)

    @lazyproperty
    def ellipticity_asymmetry(self):
        """
        The ellipticity of (the Gaussian function that has the same
        second-order moments as) the source, relative to the point that
        minimizes the asymmetry.
        """
        return self._ellipticity_generic(self._eigvals_asymmetry)

    @lazyproperty
    def elongation_asymmetry(self):
        """
        The elongation of (the Gaussian function that has the same
        second-order moments as) the source, relative to the point that
        minimizes the asymmetry.
        """
        return self._elongation_generic(self._eigvals_asymmetry)

    @lazyproperty
    def orientation_asymmetry(self):
        """
        The orientation (in radians) of the source, relative to the
        point that minimizes the asymmetry.
        """
        return self._orientation_generic(self._covariance_asymmetry)

    @lazyproperty
    def asymmetry(self):
        """
        Calculate asymmetry as described in Lotz et al. (2004).
        """
        image = self._cutout_stamp_maskzeroed
        asym = self._asymmetry_function(self._asymmetry_center,
                                        image, 'cas')
        
        return asym

    def _radius_at_fraction_of_total_cas(self, fraction):
        """
        Specialization of ``_radius_at_fraction_of_total_circ`` for
        the CAS calculations.
        """
        image = self._cutout_stamp_maskzeroed
        center = self._asymmetry_center
        r_upper = self._petro_extent_cas * self.rpetro_circ
        
        r, flag = _radius_at_fraction_of_total_circ(image, center, r_upper, fraction)
        self.flag = max(self.flag, flag)
        
        if np.isnan(r) or (r <= 0.0):
            warnings.warn('[CAS] Invalid radius_at_fraction_of_total.',
                          AstropyUserWarning)
            self.flag = 1
            r = -99.0  # invalid
        
        return r

    @lazyproperty
    def r20(self):
        """
        The radius that contains 20% of the light within
        'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
        """
        return self._radius_at_fraction_of_total_cas(0.2)

    @lazyproperty
    def r50(self):
        """
        The radius that contains 50% of the light within
        'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
        """
        return self._radius_at_fraction_of_total_cas(0.5)

    @lazyproperty
    def r80(self):
        """
        The radius that contains 80% of the light within
        'petro_extent_cas' (usually 1.5) times 'rpetro_circ'.
        """
        return self._radius_at_fraction_of_total_cas(0.8)

    @lazyproperty
    def concentration(self):
        """
        Calculate concentration as described in Lotz et al. (2004).
        """
        if (self.r20 == -99.0) or (self.r80 == -99.0):
            C = -99.0  # invalid
        else:
            C = 5.0 * np.log10(self.r80 / self.r20)
        
        return C

    @lazyproperty
    def smoothness(self):
        """
        Calculate smoothness (a.k.a. clumpiness) as described in
        Conselice (2003).
        """
        image = self._cutout_stamp_maskzeroed

        # Exclude central region during smoothness calculation:
        r_in = self._petro_fraction_cas * self.rpetro_circ
        r_out = self._petro_extent_cas * self.rpetro_circ
        ap = photutils.CircularAnnulus(self._asymmetry_center, r_in, r_out)

        boxcar_size = int(self._petro_fraction_cas * self.rpetro_circ)
        image_smooth = ndi.uniform_filter(image, size=boxcar_size)
        
        image_diff = image - image_smooth
        image_diff[image_diff < 0] = 0.0  # set negative pixels to zero

        ap_flux = ap.do_photometry(image, method='exact')[0][0]
        ap_diff = ap.do_photometry(image_diff, method='exact')[0][0]

        if ap_flux <= 0:
            warnings.warn('[smoothness] Nonpositive total flux.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid

        if self._sky_smoothness == -99.0:  # invalid skybox
            S = ap_diff / ap_flux
        else:
            S = (ap_diff - ap.area()*self._sky_smoothness) / ap_flux

        if not np.isfinite(S):
            warnings.warn('Invalid smoothness.', AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid

        return S

    ##################
    # MID statistics #
    ##################

    def _segmap_mid_main_clump(self, q):
        """
        For a given quantile ``q``, return a boolean array indicating
        the locations of pixels above ``q`` (within the original segment)
        that are also part of the "main" clump.
        """
        threshold = _quantile(self._sorted_pixelvals_stamp_no_bg_nonnegative, q)
        above_threshold = self._cutout_stamp_maskzeroed_no_bg_nonnegative >= threshold

        # Instead of assuming that the main segment is at the center
        # of the stamp, use the position of the brightest pixel:
        ic = int(self._y_maxval_stamp)
        jc = int(self._x_maxval_stamp)

        # Grow regions using 8-connected neighbor "footprint"
        s = ndi.generate_binary_structure(2, 2)
        labeled_array, num_features = ndi.label(above_threshold, structure=s)

        # Sanity check (brightest pixel should be part of the main clump):
        assert labeled_array[ic, jc] != 0

        return labeled_array == labeled_array[ic, jc]

    def _segmap_mid_function(self, q):
        """
        Helper function to calculate the MID segmap.
        
        For a given quantile ``q``, return the ratio of the mean flux of
        pixels at the level of ``q`` (within the main clump) divided by
        the mean of pixels above ``q`` (within the main clump).
        """
        locs_main_clump = self._segmap_mid_main_clump(q)
        
        mean_flux_main_clump = np.mean(
            self._cutout_stamp_maskzeroed_no_bg_nonnegative[locs_main_clump])
        mean_flux_new_pixels = _quantile(
            self._sorted_pixelvals_stamp_no_bg_nonnegative, q)

        if mean_flux_main_clump == 0:
            warnings.warn('[segmap_mid] Zero flux sum.', AstropyUserWarning)
            ratio = 1.0
            self.flag = 1
        else:
            ratio = mean_flux_new_pixels / mean_flux_main_clump

        return ratio - self._eta

    @lazyproperty
    def _segmap_mid(self):
        """
        Create a new segmentation map as described in Section 4.3 from
        Freeman et al. (2013).
        
        Notes
        -----
        This implementation improves upon previous ones by making
        the MID segmap independent from the number of quantiles
        used in the calculation, as well as other parameters.
        """
        num_pixelvals = len(self._sorted_pixelvals_stamp_no_bg_nonnegative)
        
        # In some rare cases (as a consequence of an erroneous
        # initial segmap, as in J095553.0+694048_g.fits.gz),
        # the MID segmap is technically undefined because the
        # mean flux of "newly added" pixels never reaches the
        # target value, at least within the original segmap.
        # In these cases we simply assume that the MID segmap
        # is equal to the Gini segmap.
        if self._segmap_mid_function(0.0) > 0.0:
            warnings.warn('segmap_mid is undefined; using segmap_gini instead.',
                          AstropyUserWarning)
            return self._segmap_gini
        
        # Find appropriate quantile using numerical solver
        q_min = 0.0
        q_max = 1.0
        xtol = 1.0 / float(num_pixelvals)
        q = opt.brentq(self._segmap_mid_function, q_min, q_max, xtol=xtol)

        locs_main_clump = self._segmap_mid_main_clump(q)

        # Regularize a bit the shape of the segmap:
        segmap_float = ndi.uniform_filter(
            np.float64(locs_main_clump), size=self._boxcar_size_mid)
        segmap = segmap_float > 0.5

        # Make sure that brightest pixel is in segmap
        ic = int(self._y_maxval_stamp)
        jc = int(self._x_maxval_stamp)
        if ~segmap[ic, jc]:
            warnings.warn('[segmap_mid] Adding brightest pixel to segmap.',
                          AstropyUserWarning)
            segmap[ic, jc] = True
            self.flag = 1

        # Grow regions with 8-connected neighbor "footprint"
        s = ndi.generate_binary_structure(2, 2)
        labeled_array, num_features = ndi.label(segmap, structure=s)

        return labeled_array == labeled_array[ic, jc]

    @lazyproperty
    def _cutout_mid(self):
        """
        Apply the MID segmap to the postage stamp cutout
        and set negative pixels to zero.
        """
        image = np.where(self._segmap_mid,
                         self._cutout_stamp_maskzeroed_no_bg_nonnegative, 0.0)
        return image

    @lazyproperty
    def _sorted_pixelvals_mid(self):
        """
        Just the sorted pixel values of the MID cutout.
        """
        image = self._cutout_mid
        return np.sort(image[~self._mask_stamp_no_bg])

    def _multimode_function(self, q):
        """
        Helper function to calculate the multimode statistic.
        Returns the sorted "areas" of the clumps at quantile ``q``.
        """
        threshold = _quantile(self._sorted_pixelvals_mid, q)
        above_threshold = self._cutout_mid >= threshold

        # Neighbor "footprint" for growing regions, including corners:
        s = ndi.generate_binary_structure(2, 2)

        labeled_array, num_features = ndi.label(above_threshold, structure=s)

        # Zero is reserved for non-labeled pixels:
        labeled_array_nonzero = labeled_array[labeled_array != 0]
        labels, counts = np.unique(labeled_array_nonzero, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]

        return sorted_counts

    def _multimode_ratio(self, q):
        """
        For a given quantile ``q``, return the "ratio" (A2/A1)*A2
        multiplied by -1, which is used for minimization.
        """
        invalid = np.sum(self._cutout_mid)  # high "energy" for basin-hopping
        if (q < 0) or (q > 1):
            ratio = invalid
        else:
            sorted_counts = self._multimode_function(q)
            if len(sorted_counts) == 1:
                ratio = invalid
            else:
                ratio = -1.0 * float(sorted_counts[1])**2 / float(sorted_counts[0])

        return ratio

    @lazyproperty
    def multimode(self):
        """
        Calculate the multimode (M) statistic as described in
        Freeman et al. (2013) and Peth et al. (2016).
        
        Notes
        -----
        In the original publication, Freeman et al. (2013)
        recommends using the more robust quantity (A2/A1)*A2,
        while Peth et al. (2016) recommends using the
        size-independent quantity A2/A1. Here we take a mixed
        approach (which, incidentally, is also what Mike Peth's
        IDL implementation actually does): we maximize the
        quantity (A2/A1)*A2 (as a function of the brightness
        threshold) but ultimately define the M statistic
        as the corresponding A2/A1 value.
        
        The original IDL implementation only explores quantiles
        in the range [0.5, 1.0], at least with the default settings.
        While this might be useful in practice, in theory the
        maximum (A2/A1)*A2 value could also happen in the quantile
        range [0.0, 0.5], so here we take a safer, more general
        approach and search over [0.0, 1.0].
        
        In practice, the quantity (A2/A1)*A2 is tricky to optimize.
        We improve over previous implementations by doing so
        in two stages: starting with a brute-force search
        over a relatively coarse array of quantiles, as in the
        original implementation, followed by a finer search using
        the basin-hopping method. This should do a better job of
        finding the global maximum.
        
        """
        q_min = 0.0
        q_max = 1.0

        # STAGE 1: brute-force

        # We start with a relatively coarse separation between the
        # quantiles, equal to the value used in the original IDL
        # implementation. If every calculated ratio is invalid, we
        # try a smaller size.
        mid_stepsize = 0.02

        while True:
            quantile_array = np.arange(q_min, q_max, mid_stepsize)
            ratio_array = np.zeros_like(quantile_array)
            for k, q in enumerate(quantile_array):
                ratio_array[k] = self._multimode_ratio(q)
            k_min = np.argmin(ratio_array)
            q0 = quantile_array[k_min]
            ratio_min = ratio_array[k_min]
            if ratio_min < 0:  # valid "ratios" should be negative
                break
            elif mid_stepsize < 1e-3:
                warnings.warn('[M statistic] Single clump!', AstropyUserWarning)
                # This sometimes happens when the source is a star, so the user
                # might want to discard M=0 cases, depending on the dataset.
                return 0.0
            else:
                mid_stepsize = mid_stepsize / 2.0
                warnings.warn('[M statistic] Reduced stepsize to %g.' % (
                              mid_stepsize), AstropyUserWarning)

        # STAGE 2: basin-hopping method

        # The results seem quite robust to changes in this parameter,
        # so I leave it hardcoded for now:
        mid_bh_rel_temp = 0.5

        temperature = -1.0 * mid_bh_rel_temp * ratio_min
        res = opt.basinhopping(self._multimode_ratio, q0,
            minimizer_kwargs={"method": "Nelder-Mead"},
            niter=self._niter_bh_mid, T=temperature, stepsize=mid_stepsize,
            interval=self._niter_bh_mid/2, disp=False, seed=0)
        q_final = res.x[0]

        # Finally, return A2/A1 instead of (A2/A1)*A2
        sorted_counts = self._multimode_function(q_final)

        return float(sorted_counts[1]) / float(sorted_counts[0])

    @lazyproperty
    def _cutout_mid_smooth(self):
        """
        Just a Gaussian-smoothed version of the zero-masked image used
        in the MID calculations.
        """
        image_smooth = ndi.gaussian_filter(self._cutout_mid, self._sigma_mid)
        return image_smooth

    @lazyproperty
    def _watershed_mid(self):
        """
        This replaces the "i_clump" routine from the original IDL code.
        The main difference is that we do not place a limit on the
        number of labeled regions (previously limited to 100 regions).
        This is also much faster, thanks to the highly optimized
        "peak_local_max" and "watershed" skimage routines.
        Returns a labeled array indicating regions around local maxima.
        """
        peaks = skimage.feature.peak_local_max(
            self._cutout_mid_smooth, indices=True, num_peaks=np.inf)
        num_peaks = peaks.shape[0]
        # The zero label is reserved for the background:
        peak_labels = np.arange(1, num_peaks+1, dtype=np.int64)
        ypeak, xpeak = peaks.T

        markers = np.zeros(self._cutout_mid_smooth.shape, dtype=np.int64)
        markers[ypeak, xpeak] = peak_labels

        mask = self._cutout_mid_smooth > 0
        labeled_array = skimage.morphology.watershed(
            -self._cutout_mid_smooth, markers, connectivity=2, mask=mask)

        return labeled_array, peak_labels, xpeak, ypeak

    @lazyproperty
    def _intensity_sums(self):
        """
        Helper function to calculate the intensity (I) and
        deviation (D) statistics.
        """
        labeled_array, peak_labels, xpeak, ypeak = self._watershed_mid
        num_peaks = len(peak_labels)
        
        flux_sums = np.zeros(num_peaks, dtype=np.float64)
        for k, label in enumerate(peak_labels):
            locs = labeled_array == label
            flux_sums[k] = np.sum(self._cutout_mid_smooth[locs])
        sid = np.argsort(flux_sums)[::-1]
        sorted_flux_sums = flux_sums[sid]
        sorted_xpeak = xpeak[sid]
        sorted_ypeak = ypeak[sid]
        
        return sorted_flux_sums, sorted_xpeak, sorted_ypeak

    @lazyproperty
    def intensity(self):
        """
        Calculate the intensity (I) statistic as described in
        Peth et al. (2016).
        """
        sorted_flux_sums, sorted_xpeak, sorted_ypeak = self._intensity_sums
        if len(sorted_flux_sums) <= 1:
            # Unlike the M=0 cases, there seem to be some legitimate
            # I=0 cases, so we do not turn on the "bad measurement" flag.
            return 0.0
        else:
            return sorted_flux_sums[1] / sorted_flux_sums[0]

    @lazyproperty
    def deviation(self):
        """
        Calculate the deviation (D) statistic as described in
        Peth et al. (2016).
        """
        image = np.float64(self._cutout_mid)  # skimage wants double
        
        sorted_flux_sums, sorted_xpeak, sorted_ypeak = self._intensity_sums
        if len(sorted_flux_sums) == 0:
            warnings.warn('[deviation] There are no peaks.', AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid

        xp = sorted_xpeak[0] + 0.5  # center of pixel
        yp = sorted_ypeak[0] + 0.5
        
        # Calculate centroid
        M = skimage.measure.moments(image, order=1)
        if M[0, 0] <= 0:
            warnings.warn('[deviation] Nonpositive flux within MID segmap.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid
        yc = M[1, 0] / M[0, 0]
        xc = M[0, 1] / M[0, 0]
        yc += 0.5; xc += 0.5  # shift pixel positions

        area = np.sum(self._segmap_mid)
        D = np.sqrt(np.pi/area) * np.sqrt((xp-xc)**2 + (yp-yc)**2)

        if not np.isfinite(D):
            warnings.warn('Invalid D-statistic.', AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid

        return D

    ###################
    # SHAPE ASYMMETRY #
    ###################

    @lazyproperty
    def rhalf_circ(self):
        """
        The radius of a circular aperture containing 50% of the light,
        assuming that the center is the point that minimizes the
        asymmetry and that the total is at ``rmax_circ``.
        """
        image = self._cutout_stamp_maskzeroed
        center = self._asymmetry_center

        if self.rmax_circ == 0:
            r = 0.0
        else:
            r, flag = _radius_at_fraction_of_total_circ(
                image, center, self.rmax_circ, 0.5)
            self.flag = max(self.flag, flag)
        
        # In theory, this return value can also be NaN
        return r

    @lazyproperty
    def rhalf_ellip(self):
        """
        The semimajor axis of an elliptical aperture containing 50% of
        the light, assuming that the center is the point that minimizes
        the asymmetry and that the total is at ``rmax_ellip``.
        """
        image = self._cutout_stamp_maskzeroed
        center = self._asymmetry_center

        if self.rmax_ellip == 0:
            r = 0.0
        else:
            r, flag = _radius_at_fraction_of_total_ellip(
                image, center, self.elongation_asymmetry,
                self.orientation_asymmetry, self.rmax_ellip, 0.5)
            self.flag = max(self.flag, flag)
        
        # In theory, this return value can also be NaN
        return r

    @lazyproperty
    def _segmap_shape_asym(self):
        """
        Construct a binary detection mask as described in Section 3.1
        from Pawlik et al. (2016).
        
        Notes
        -----
        The original algorithm assumes that a circular aperture with
        inner and outer radii equal to 20 and 40 times the FWHM is
        representative of the background. In order to avoid amplifying
        uncertainties from the central regions of the galaxy profile,
        we assume that such an aperture has inner and outer radii
        equal to multiples of the elliptical Petrosian radius
        (which can be specified by the user). Also, we define the center
        as the point that minimizes the asymmetry, instead of the
        brightest pixel.

        """
        ny, nx = self._cutout_stamp_maskzeroed.shape

        # Center at pixel that minimizes asymmetry
        center = self._asymmetry_center

        # Create a circular annulus around the center
        # that only contains background sky (hopefully).
        r_in = self._petro_extent_flux * self.rpetro_ellip
        r_out = 2.0 * self._petro_extent_flux * self.rpetro_ellip
        circ_annulus = photutils.CircularAnnulus(center, r_in, r_out)

        # Convert circular annulus aperture to binary mask
        circ_annulus_mask = circ_annulus.to_mask(method='center')[0]
        # With the same shape as the postage stamp
        circ_annulus_mask = circ_annulus_mask.to_image((ny, nx))
        # Invert mask and exclude other sources
        total_mask = self._mask_stamp | np.logical_not(circ_annulus_mask)

        # If sky area is too small (e.g., if annulus is outside the
        # image), use skybox instead.
        if np.sum(~total_mask) < self._skybox_size**2:
            warnings.warn('[shape_asym] Using skybox for background.',
                          AstropyUserWarning)
            total_mask = np.ones((ny, nx), dtype=np.bool8)
            total_mask[self._slice_skybox] = False
            # However, if skybox is undefined, there is nothing to do.
            if np.sum(~total_mask) == 0:
                warnings.warn('[shape_asym] Asymmetry segmap undefined.',
                              AstropyUserWarning)
                self.flag = 1
                return ~self._mask_stamp_no_bg

        # Do sigma-clipping until convergence
        mean, median, std = sigma_clipped_stats(
            self._cutout_stamp_maskzeroed, mask=total_mask, sigma=3.0,
            iters=None, cenfunc=_mode)

        # Mode as defined in Bertin & Arnouts (1996)
        mode = 2.5*median - 1.5*mean
        threshold = mode + std

        # Smooth image slightly and apply 1-sigma threshold
        image_smooth = ndi.uniform_filter(
            self._cutout_stamp_maskzeroed, size=self._boxcar_size_shape_asym)
        above_threshold = image_smooth >= threshold

        # Make sure that brightest pixel (of smoothed image) is in segmap
        ic, jc = np.argwhere(image_smooth == np.max(image_smooth))[0]
        if ~above_threshold[ic, jc]:
            warnings.warn('[shape_asym] Adding brightest pixel to segmap.',
                          AstropyUserWarning)
            above_threshold[ic, jc] = True
            self.flag = 1

        # Grow regions with 8-connected neighbor "footprint"
        s = ndi.generate_binary_structure(2, 2)
        labeled_array, num_features = ndi.label(above_threshold, structure=s)

        return labeled_array == labeled_array[ic, jc]

    @lazyproperty
    def rmax_circ(self):
        """
        Return the distance (in pixels) from the pixel that minimizes
        the asymmetry to the edge of the main source segment, similar 
        to Pawlik et al. (2016).
        """
        image = self._cutout_stamp_maskzeroed
        ny, nx = image.shape

        # Center at pixel that minimizes asymmetry
        xc, yc = self._asymmetry_center

        # Distances from all pixels to the center
        ypos, xpos = np.mgrid[0:ny, 0:nx] + 0.5  # center of pixel
        distances = np.sqrt((ypos-yc)**2 + (xpos-xc)**2)
        
        # Only consider pixels within the segmap.
        rmax_circ = np.max(distances[self._segmap_shape_asym])
        
        if rmax_circ == 0:
            warnings.warn('[rmax_circ] rmax_circ = 0!', AstropyUserWarning)
            self.flag = 1
        
        return rmax_circ

    @lazyproperty
    def rmax_ellip(self):
        """
        Return the semimajor axis of the minimal ellipse (with fixed
        center, elongation and orientation) that contains all of
        the main segment of the shape asymmetry segmap. In most
        cases this is almost identical to rmax_circ.
        """
        image = self._cutout_stamp_maskzeroed
        ny, nx = image.shape

        # Center at pixel that minimizes asymmetry
        xc, yc = self._asymmetry_center

        theta = self.orientation_asymmetry
        y, x = np.mgrid[0:ny, 0:nx] + 0.5  # center of pixel

        xprime = (x-xc)*np.cos(theta) + (y-yc)*np.sin(theta)
        yprime = -(x-xc)*np.sin(theta) + (y-yc)*np.cos(theta)
        r_ellip = np.sqrt(xprime**2 + (yprime*self.elongation_asymmetry)**2)

        # Only consider pixels within the segmap.
        rmax_ellip = np.max(r_ellip[self._segmap_shape_asym])
        
        if rmax_ellip == 0:
            warnings.warn('[rmax_ellip] rmax_ellip = 0!', AstropyUserWarning)
            self.flag = 1
        
        return rmax_ellip

    @lazyproperty
    def outer_asymmetry(self):
        """
        Calculate outer asymmetry as described in Wen et al. (2014).
        Note that the center is the one used for the standard asymmetry.
        """
        image = self._cutout_stamp_maskzeroed
        asym = self._asymmetry_function(self._asymmetry_center, image, 'outer')
        
        return asym

    @lazyproperty
    def shape_asymmetry(self):
        """
        Calculate shape asymmetry as described in Pawlik et al. (2016).
        Note that the center is the one used for the standard asymmetry.
        """
        image = np.where(self._segmap_shape_asym, 1.0, 0.0)
        asym = self._asymmetry_function(self._asymmetry_center, image, 'shape')

        return asym

    ####################
    # SERSIC MODEL FIT #
    ####################

    @lazyproperty
    def _sersic_model(self):
        """
        Fit a 2D Sersic profile using Astropy's model fitting library.
        Return the fitted model object.
        """
        image = self._cutout_stamp_maskzeroed
        ny, nx = image.shape
        center = self._asymmetry_center
        theta = self.orientation_asymmetry

        # Get flux at the "effective radius"
        a_in = self.rhalf_ellip - 0.5 * self._annulus_width
        a_out = self.rhalf_ellip + 0.5 * self._annulus_width
        if a_in < 0:
            warnings.warn('[sersic] rhalf_ellip < annulus_width.',
                          AstropyUserWarning)
            self.flag_sersic = 1
            a_in = self.rhalf_ellip
        b_out = a_out / self.elongation_asymmetry
        ellip_annulus = photutils.EllipticalAnnulus(
            center, a_in, a_out, b_out, theta)
        ellip_annulus_mean_flux = _aperture_mean_nomask(
            ellip_annulus, image, method='exact')
        if ellip_annulus_mean_flux <= 0.0:
            warnings.warn('[sersic] Nonpositive flux at r_e.', AstropyUserWarning)
            self.flag_sersic = 1
            ellip_annulus_mean_flux = np.abs(ellip_annulus_mean_flux)

        # Prepare data for fitting
        z = image.copy()
        y, x = np.mgrid[0:ny, 0:nx] + 0.5  # center of pixel
        weightmap = self._weightmap_stamp
        # Exclude pixels with image == 0 or weightmap == 0 from the fit.
        fit_weights = np.zeros_like(z)
        locs = (image != 0) & (weightmap != 0)
        # The sky background noise is already included in the weightmap:
        fit_weights[locs] = 1.0 / weightmap[locs]

        # Initial guess
        if self.concentration < 3.0:
            guess_n = 1.0
        elif self.concentration < 4.0:
            guess_n = 2.0
        else:
            guess_n = 3.5
        xc, yc = self._asymmetry_center
        if self._psf is None:
            sersic_init = models.Sersic2D(
                amplitude=ellip_annulus_mean_flux, r_eff=self.rhalf_ellip,
                n=guess_n, x_0=xc, y_0=yc, ellip=self.ellipticity_asymmetry, theta=theta)
        else:
            sersic_init = ConvolvedSersic2D(
                amplitude=ellip_annulus_mean_flux, r_eff=self.rhalf_ellip,
                n=guess_n, x_0=xc, y_0=yc, ellip=self.ellipticity_asymmetry, theta=theta)
            sersic_init.set_psf(self._psf)

        # The number of data points cannot be smaller than the number of
        # free parameters (7 in the case of Sersic2D)
        if z.size < sersic_init.parameters.size:
            warnings.warn('[sersic] Not enough data for fit.',
                          AstropyUserWarning)
            self.flag_sersic = 1
            return sersic_init

        # Since model fitting can be computationally expensive (especially
        # with a large PSF), only do it when the other measurements are OK.
        if self.flag == 1:
            warnings.warn('[sersic] Skipping Sersic fit...',
                          AstropyUserWarning)
            self.flag_sersic = 1
            return sersic_init

        # Try to fit model
        fit_sersic = fitting.LevMarLSQFitter()
        sersic_model = fit_sersic(sersic_init, x, y, z, weights=fit_weights,
                                  maxiter=self._sersic_maxiter, acc=1e-5)
        if fit_sersic.fit_info['ierr'] not in [1, 2, 3, 4]:
            warnings.warn("fit_info['message']: " + fit_sersic.fit_info['message'],
                          AstropyUserWarning)
            self.flag_sersic = 1

        return sersic_model

    @lazyproperty
    def sersic_amplitude(self):
        """
        The amplitude of the 2D Sersic fit at the effective (half-light)
        radius (`astropy.modeling.models.Sersic2D`).
        """
        return self._sersic_model.amplitude.value

    @lazyproperty
    def sersic_rhalf(self):
        """
        The effective (half-light) radius of the 2D Sersic fit
        (`astropy.modeling.models.Sersic2D`).
        """
        return self._sersic_model.r_eff.value

    @lazyproperty
    def sersic_n(self):
        """
        The Sersic index ``n`` (`astropy.modeling.models.Sersic2D`).
        """
        return self._sersic_model.n.value

    @lazyproperty
    def sersic_xc(self):
        """
        The x-coordinate of the center of the 2D Sersic fit
        (`astropy.modeling.models.Sersic2D`), relative to the
        original image.
        """
        return self.xmin_stamp + self._sersic_model.x_0.value

    @lazyproperty
    def sersic_yc(self):
        """
        The y-coordinate of the center of the 2D Sersic fit
        (`astropy.modeling.models.Sersic2D`), relative to the
        original image.
        """
        return self.ymin_stamp + self._sersic_model.y_0.value

    @lazyproperty
    def sersic_ellip(self):
        """
        The ellipticity of the 2D Sersic fit
        (`astropy.modeling.models.Sersic2D`).
        """
        return self._sersic_model.ellip.value

    @lazyproperty
    def sersic_theta(self):
        """
        The orientation (counterclockwise, in radians) of the
        2D Sersic fit (`astropy.modeling.models.Sersic2D`).
        """
        theta = self._sersic_model.theta.value
        return theta - np.floor(theta/np.pi) * np.pi


def source_morphology(image, segmap, **kwargs):
    """
    Calculate the morphological parameters of all sources in ``image``
    as labeled by ``segmap``.
    
    Parameters
    ----------
    image : array-like
        A 2D image containing the sources of interest.
        The image must already be background-subtracted.
    segmap : array-like (int) or `photutils.SegmentationImage`
        A 2D segmentation map where different sources are 
        labeled with different positive integer values.
        A value of zero is reserved for the background.

    Other parameters
    ----------------
    kwargs : `~statmorph.SourceMorphology` properties.

    Returns
    -------
    sources_morph : list
        A list of `SourceMorphology` objects, one for each
        source. The morphological parameters can be accessed
        as attributes or keys.

    See Also
    --------
    `SourceMorphology` : Class to measure morphological parameters.
    
    Examples
    --------
    See `README.rst` for usage examples.
    
    References
    ----------
    See `README.rst` for a list of references.

    """
    if not isinstance(segmap, photutils.SegmentationImage):
        segmap = photutils.SegmentationImage(segmap)

    sources_morph = []
    for label in segmap.labels:
        sources_morph.append(SourceMorphology(image, segmap, label, **kwargs))
        print('Finished processing source %d.\n' % (label))

    return sources_morph
