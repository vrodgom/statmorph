"""
Python code for calculating non-parametric morphological diagnostics
of galaxy images.
"""
# Author: Vicente Rodriguez-Gomez <vrg@jhu.edu>
# Licensed under a 3-Clause BSD License.

import numpy as np
import time
import scipy.optimize as opt
import scipy.ndimage as ndi
import skimage.measure
import skimage.feature
import skimage.morphology
from astropy.utils import lazyproperty
from astropy.stats import sigma_clipped_stats
import photutils

__all__ = ['SourceMorphology', 'source_morphology']

def _quantile(sorted_values, q):
    """
    For a sorted (in increasing order) 1-d array, return the
    value corresponding to the quantile ``q``.
    """
    assert((q >= 0) and (q <= 1))
    if q == 1:
        return sorted_values[-1]
    else:
        return sorted_values[int(q*len(sorted_values))]

def _mode(a):
    """
    Takes a masked array as input and returns the "mode"
    as defined in Bertin & Arnouts (1996):
    mode = 2.5 * median - 1.5 * mean
    """
    return 2.5 * np.ma.median(a) - 1.5 * np.ma.mean(a)

def _fraction_of_total_function(r, image, center, fraction, total_sum):
    """
    Helper function to calculate ``_radius_at_fraction_of_total``.
    The ``center`` is given as (x,y)
    """
    ap = photutils.CircularAperture(center, r)
    ap_sum = ap.do_photometry(image, method='exact')[0][0]

    return ap_sum / total_sum - fraction

def _radius_at_fraction_of_total(image, center, r_max, fraction):
    """
    Return the radius (in pixels) of a concentric circle
    that contains a given fraction of the total light.
    The ``center`` is given as (x,y)
    """
    r_min = 1.0
    ap_total = photutils.CircularAperture(center, r_max)
    total_sum = ap_total.do_photometry(image, method='exact')[0][0]
    r = opt.brentq(_fraction_of_total_function, r_min, r_max,
                   args=(image, center, fraction, total_sum), xtol=1e-6)

    return r

def _fraction_of_maximum_function(r, image, mask, center, annulus_width, fraction, max_flux):
    """
    Helper function to calculate ``_radius_at_fraction_of_maximum``.
    """
    r_in = r - 0.5 * annulus_width
    r_out = r + 0.5 * annulus_width

    circ_annulus = photutils.CircularAnnulus(center, r_in, r_out)
    circ_annulus_mean_flux = _aperture_mean(
        circ_annulus, image, mask, method='exact')

    return circ_annulus_mean_flux / max_flux - fraction

def _radius_at_fraction_of_maximum(image, mask, r_max, annulus_width, fraction):
    """
    Return the radius at which the mean flux is a given
    fraction of the maximum.
    """
    r_min = 1.0
    max_flux = np.max(image)
    yc, xc = np.argwhere(image == max_flux)[0]
    center = np.array([xc, yc]) + 0.5
    r = opt.brentq(_fraction_of_maximum_function, r_min, r_max, xtol=1e-6,
                   args=(image, mask, center, annulus_width, fraction, max_flux))

    return r

# The photutils aperture photometry functions *almost* do what I want,
# but not quite, so unfortunately we need the following:

def _aperture_area(ap, mask, **kwargs):
    """
    Calculate the area of a photutils aperture object,
    excluding masked pixels.
    """
    return ap.do_photometry(np.float64(~mask), **kwargs)[0][0]

def _aperture_mean(ap, image, mask, **kwargs):
    """
    Calculate the mean flux of an image for a given photutils
    aperture object and a mask.
    """
    image2 = image.copy()  # because I don't trust Python sometimes
    image2[mask] = 0.0
    flux_sum = ap.do_photometry(image2, **kwargs)[0][0]

    return flux_sum / _aperture_area(ap, mask, **kwargs)

def _aperture_std(ap, image, mask, **kwargs):
    """
    Calculate the standard deviation of an image for a given
    photutils aperture object and a mask.
    """
    sqr_diff = (image - _aperture_mean(ap, image, mask, **kwargs))**2
    sqr_diff[mask] = 0.0
    sqr_diff_sum = ap.do_photometry(sqr_diff, **kwargs)[0][0]

    return np.sqrt(sqr_diff_sum / _aperture_area(ap, mask, **kwargs))

# This is the main morphology class:

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
    label : int
        A label indicating the source of interest.
    mask : array-like (bool), optional
        A 2D array with the same size as ``image``, where pixels
        set to ``True`` are ignored from all calculations.
    variance : array-like, optional
        A 2D array with the same size as ``image`` with the
        values of the variance (RMS^2). This is required
        in order to calculate the signal-to-noise correctly.
    cutout_extent : float, optional
        The target fractional size of the data cutout relative to
        the minimal bounding box containing the source. The value
        must be >= 1. The default value is 2.0 (i.e., 2 times larger).
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
    remove_outliers : bool, optional
        If ``True``, remove outlying pixels as described in Lotz et al.
        (2004), using the parameter ``n_sigma_outlier``. This is the
        most time-consuming operation and, at least for reasonably
        clean data, it should have a negligible effect on Gini
        coefficient. By default it is set to ``False``.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a pixel as an
        outlier, relative to its 8 neighbors. This parameter only
        takes effect when ``remove_outliers`` is ``True``. The default
        value is 10.
    border_size : scalar, optional
        The number of pixels that are skipped from each border of the
        "postage stamp" image cutout when finding the skybox. The
        default is 5 pixels.
    skybox_size : scalar, optional
        The size in pixels of the (square) "skybox" used to measure
        properties of the image background. The default is 20 pixels.
    petro_extent : float, optional
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
        iterations in the basin-hopping stage of the maximization. The
        default value of 100 is probably enough for "production" runs,
        but lower values can significantly increase speed.
    sigma_mid : float, optional
        In the MID calculations, this is the smoothing scale (in pixels)
        used to compute the intensity (I) statistic. The default is 1.0.
    sky_num_fwhm : scalar, optional
        When calculating the shape asymmetry segmap, it is assumed that
        a circular annulus with inner and outer *diameter* equal to
        ``num_fwhm`` and 2*``num_fwhm`` times the FWHM of the galaxy
        profile is representative of the background. The default is 20.
    boxcar_size_shape_asym : float, optional
        When calculating the shape asymmetry segmap, this is the size
        (in pixels) of the constant kernel used to regularize the segmap.
        The default value is 3.0.

    References
    ----------
    See `README.md` for a list of references.

    """
    def __init__(self, image, segmap, label, mask=None, variance=None,
                 cutout_extent=2.0, annulus_width=1.0, eta=0.2,
                 petro_fraction_gini=0.2, remove_outliers=False,
                 n_sigma_outlier=10, border_size=5, skybox_size=20,
                 petro_extent=1.5, petro_fraction_cas=0.25,
                 boxcar_size_mid=3.0, niter_bh_mid=100, sigma_mid=1.0,
                 sky_num_fwhm=20.0, boxcar_size_shape_asym=3.0):
        self._variance = variance
        self._cutout_extent = cutout_extent
        self._annulus_width = annulus_width
        self._eta = eta
        self._petro_fraction_gini = petro_fraction_gini
        self._remove_outliers = remove_outliers
        self._n_sigma_outlier = n_sigma_outlier
        self._border_size = border_size
        self._skybox_size = skybox_size
        self._petro_extent = petro_extent
        self._petro_fraction_cas = petro_fraction_cas
        self._boxcar_size_mid = boxcar_size_mid
        self._niter_bh_mid = niter_bh_mid
        self._sigma_mid = sigma_mid
        self._sky_num_fwhm = sky_num_fwhm
        self._boxcar_size_shape_asym = boxcar_size_shape_asym

        # The following object stores some important data:
        self._props = photutils.SourceProperties(image, segmap, label, mask=mask)

        # Centroid of the source relative to the "postage stamp" cutout:
        self._xc_stamp = self._props.xcentroid.value - self._slice_stamp[1].start
        self._yc_stamp = self._props.ycentroid.value - self._slice_stamp[0].start

        # Position of the brightest pixel relative to the cutout:
        self._x_maxval_stamp = self._props.maxval_xpos.value - self._slice_stamp[1].start
        self._y_maxval_stamp = self._props.maxval_ypos.value - self._slice_stamp[0].start

    def __getitem__(self, key):
        return getattr(self, key)

    @lazyproperty
    def _slice_stamp(self):
        """
        Attempt to create a square slice (centered at the centroid)
        that is a bit larger than the minimal bounding box containing
        the main labeled segment. Note that the cutout may not be
        square when the source is close to a border of the original image.
        """
        # Maximum distance to any side of the bounding box
        yc, xc = np.int64(self._props.centroid.value)
        ymin, xmin, ymax, xmax = np.int64(self._props.bbox.value)
        dist = max(xmax-xc, xc-xmin, ymax-yc, yc-ymin)

        # Add some extra space in each dimension
        assert(self._cutout_extent >= 1.0)
        dist = int(dist * self._cutout_extent)

        # Make cutout
        ny, nx = self._props._data.shape
        slice_stamp = (slice(max(0, yc-dist), min(ny, yc+dist)),
                        slice(max(0, xc-dist), min(nx, xc+dist)))

        return slice_stamp

    @lazyproperty
    def _mask_stamp(self):
        """
        Create a total binary mask for the "postage stamp".
        Pixels belonging to other sources (as well as pixels masked
        using the ``mask`` keyword argument) are set to ``True``,
        but the background (segmap == 0) is left alone.
        """
        segmap_stamp = self._props._segment_img.data[self._slice_stamp]
        mask_stamp = (segmap_stamp != 0) & (segmap_stamp != self._props.label)
        if self._props._mask is not None:
            mask_stamp = mask_stamp | self._props._mask[self._slice_stamp]
        return mask_stamp

    @lazyproperty
    def _cutout_stamp_maskzeroed(self):
        """
        Return a data cutout with its shape and position determined
        by ``_slice_stamp``. Pixels belonging to other sources
        (as well as pixels where ``mask`` == 1) are set to zero,
        but the background is left alone.
        """
        cutout_stamp = self._props._data[self._slice_stamp]
        cutout_stamp[self._mask_stamp] = 0

        return cutout_stamp

    @lazyproperty
    def _sorted_pixelvals_stamp(self):
        """
        Just the sorted pixel values of the postage stamp.
        """
        return np.sort(self._cutout_stamp_maskzeroed.flatten())

    @lazyproperty
    def _dist_to_closest_corner(self):
        """
        The distance from the centroid to the closest corner of the
        minimal bounding box containing the source. This is used as an
        upper limit when computing the Petrosian radius.
        """
        x_dist = min(self._props.xmax.value - self._props.xcentroid.value,
                     self._props.xcentroid.value - self._props.xmin.value)
        y_dist = min(self._props.ymax.value - self._props.ycentroid.value,
                     self._props.ycentroid.value - self._props.ymin.value)
        return np.sqrt(x_dist**2 + y_dist**2)

    def _petrosian_function_ellip(self, a):
        """
        Helper function to calculate the Petrosian "radius".
        
        For the ellipse with semi-major axis ``a``, return the
        ratio of the mean flux over an elliptical annulus
        divided by the mean flux within the ellipse,
        minus "eta" (eq. 4 from Lotz et al. 2004). The root of
        this function is the Petrosian "radius".
        """
        b = a / self._props.elongation.value
        a_in = a - 0.5 * self._annulus_width
        a_out = a + 0.5 * self._annulus_width

        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value

        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_stamp, self._yc_stamp), a_in, a_out, b_out, theta)
        ellip_aperture = photutils.EllipticalAperture(
            (self._xc_stamp, self._yc_stamp), a, b, theta)

        ellip_annulus_mean_flux = _aperture_mean(
            ellip_annulus, self._cutout_stamp_maskzeroed,
            self._mask_stamp, method='exact')
        ellip_aperture_mean_flux = _aperture_mean(
            ellip_aperture, self._cutout_stamp_maskzeroed,
            self._mask_stamp, method='exact')

        return ellip_annulus_mean_flux / ellip_aperture_mean_flux - self._eta

    def _petrosian_function_circ(self, r):
        """
        Helper function to calculate ``petrosian_radius_circ``.
        
        For the circle with radius `r`, return the
        ratio of the mean flux over a pixel-wide circular
        annulus divided by the mean flux within the circle,
        minus "eta" (eq. 4 from Lotz et al. 2004). The root of
        this function is the Petrosian radius.
        """
        r_in = r - 0.5 * self._annulus_width
        r_out = r + 0.5 * self._annulus_width

        circ_annulus = photutils.CircularAnnulus(
            (self._xc_stamp, self._yc_stamp), r_in, r_out)
        circ_aperture = photutils.CircularAperture(
            (self._xc_stamp, self._yc_stamp), r)

        circ_annulus_mean_flux = _aperture_mean(
            circ_annulus, self._cutout_stamp_maskzeroed,
            self._mask_stamp, method='exact')
        circ_aperture_mean_flux = _aperture_mean(
            circ_aperture, self._cutout_stamp_maskzeroed,
            self._mask_stamp, method='exact')

        return circ_annulus_mean_flux / circ_aperture_mean_flux - self._eta

    @lazyproperty
    def petrosian_radius_ellip(self):
        """
        Compute the Petrosian "radius" (actually the semi-major axis)
        for concentric elliptical apertures.
        """
        a_min = self._annulus_width
        a_max = self._dist_to_closest_corner
        rpetro_ellip = opt.brentq(self._petrosian_function_ellip,
                                  a_min, a_max, xtol=1e-6)
        
        return rpetro_ellip

    @lazyproperty
    def petrosian_radius_circ(self):
        """
        Compute the Petrosian radius for concentric circular apertures.
        """
        r_min = self._annulus_width
        r_max = self._dist_to_closest_corner
        rpetro_circ = opt.brentq(self._petrosian_function_circ,
                                 r_min, r_max, xtol=1e-6)

        return rpetro_circ

    #######################
    # Gini-M20 statistics #
    #######################

    @lazyproperty
    def _cutout_gini(self):
        """
        Remove outliers as described in Lotz et al. (2004).
        """
        if self._remove_outliers:
            # For performance checks
            start = time.time()

            # Exclude the center pixel from calculations
            local_footprint = np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ])
            local_mean = ndi.filters.generic_filter(
                self._cutout_stamp_maskzeroed, np.mean, footprint=local_footprint)
            local_std = ndi.filters.generic_filter(
                self._cutout_stamp_maskzeroed, np.std, footprint=local_footprint)
            bad_pixels = (np.abs(self._cutout_stamp_maskzeroed - local_mean) >
                          self._n_sigma_outlier * local_std)
            cutout_gini = np.where(~bad_pixels, self._cutout_stamp_maskzeroed, 0)
            
            print('There are %d bad pixels.' % (np.sum(bad_pixels)))
            print('It took', time.time() - start, 's to remove them.')

        else:
            cutout_gini = self._cutout_stamp_maskzeroed

        return cutout_gini

    @lazyproperty
    def _segmap_gini(self):
        """
        Create a new segmentation map (relative to the "Gini" cutout)
        based on the Petrosian "radius".
        """
        # Smooth image
        petro_sigma = self._petro_fraction_gini * self.petrosian_radius_ellip
        cutout_smooth = ndi.gaussian_filter(self._cutout_gini, petro_sigma)

        # Use mean flux at the Petrosian "radius" as threshold
        a_in = self.petrosian_radius_ellip - 0.5 * self._annulus_width
        a_out = self.petrosian_radius_ellip + 0.5 * self._annulus_width
        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value
        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_stamp, self._yc_stamp), a_in, a_out, b_out, theta)
        ellip_annulus_mean_flux = _aperture_mean(
            ellip_annulus, cutout_smooth, self._mask_stamp, method='exact')
        
        return cutout_smooth >= ellip_annulus_mean_flux

    @lazyproperty
    def gini(self):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        """
        image = self._cutout_gini.flatten()
        segmap = self._segmap_gini.flatten()

        sorted_pixelvals = np.sort(np.abs(image[segmap]))

        n = len(sorted_pixelvals)
        indices = np.arange(1, n+1)  # start at i=1
        gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
                (float(n-1) * np.sum(sorted_pixelvals)))

        return gini

    @lazyproperty
    def m20(self):
        """
        Calculate the M_20 coefficient as described in Lotz et al. (2004).
        """
        # Use the same region as in the Gini calculation
        image = np.where(self._segmap_gini, self._cutout_gini, 0.0)
        image = np.float64(image)  # skimage wants double

        # Calculate centroid
        m = skimage.measure.moments(image, order=1)
        yc = m[0, 1] / m[0, 0]
        xc = m[1, 0] / m[0, 0]

        # Calculate second total central moment
        mc = skimage.measure.moments_central(image, yc, xc, order=3)
        second_moment = mc[0, 2] + mc[2, 0]

        # Calculate threshold pixel value
        sorted_pixelvals = np.sort(image.flatten())
        flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
        threshold = sorted_pixelvals[flux_fraction >= 0.8][0]

        # Calculate second moment of the brightest pixels
        image_20 = np.where(image >= threshold, image, 0.0)
        mc_20 = skimage.measure.moments_central(image_20, yc, xc, order=3)
        second_moment_20 = mc_20[0, 2] + mc_20[2, 0]

        # Normalize
        m20 = np.log10(second_moment_20 / second_moment)

        return m20

    @lazyproperty
    def sn_per_pixel(self):
        """
        Calculate the signal-to-noise per pixel using the Petrosian segmap.
        """
        image = self._cutout_gini
        locs = self._segmap_gini & (image >= 0)
        pixelvals = image[locs]
        if self._variance is None:
            variance = np.zeros_like(pixelvals)
        else:
            variance = self._variance[self._slice_stamp]

        return np.mean(image[locs] / np.sqrt(variance[locs] +
                                             self._sky_sigma**2))

    ##################
    # CAS statistics #
    ##################

    @lazyproperty
    def _slice_skybox(self):
        """
        Find a region of the sky that only contains background.
        
        In principle, a more accurate approach could be adopted
        (e.g. Shi et al. 2009, ApJ, 697, 1764).
        """
        border_size = self._border_size
        skybox_size = self._skybox_size

        image = self._props._data[self._slice_stamp]
        segmap = self._props._segment_img.data[self._slice_stamp]
        mask = np.zeros(image.shape, dtype=np.bool8)
        if self._props._mask is not None:
            mask = self._props._mask[self._slice_stamp]

        ny, nx = image.shape
        for i in range(border_size, ny - border_size - skybox_size):
            for j in range(border_size, nx - border_size - skybox_size):
                boxslice = (slice(i, i + skybox_size),
                            slice(j, j + skybox_size))
                if np.all(segmap[boxslice] == 0) and np.all(~mask[boxslice]):
                    return boxslice

        # If we got here, something went wrong.
        raise Exception('Error: skybox not found.')

    @lazyproperty
    def _sky_mean(self):
        """
        Standard deviation of the background.
        """
        return np.mean(self._cutout_stamp_maskzeroed[self._slice_skybox])

    @lazyproperty
    def _sky_sigma(self):
        """
        Mean background value.
        """
        return np.std(self._cutout_stamp_maskzeroed[self._slice_skybox])

    @lazyproperty
    def _sky_asymmetry(self):
        """
        Asymmetry of the background. Note the peculiar normalization.
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]
        bkg_180 = bkg[::-1, ::-1]
        return np.sum(np.abs(bkg_180 - bkg)) / float(bkg.size)

    @lazyproperty
    def _sky_smoothness(self):
        """
        Smoothness of the background. Note the peculiar normalization.
        """
        bkg = self._cutout_stamp_maskzeroed[self._slice_skybox]

        # If the smoothing "boxcar" is larger than the skybox itself,
        # this just sets all values equal to the mean:
        boxcar_size = int(self._petro_fraction_cas * self.petrosian_radius_circ)
        bkg_smooth = ndi.uniform_filter(bkg, size=boxcar_size)

        return np.sum(np.abs(bkg_smooth - bkg)) / float(bkg.size)

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
        
        Notes
        -----
        Here are some notes about why I'm *not* using
        skimage.transform.rotate. The following line would
        work correctly in skimage version 0.13.0:
        skimage.transform.rotate(image, 180.0, center=np.floor(center))
        However, note that the "center" argument must be truncated,
        which is not mentioned in the documentation
        (https://github.com/scikit-image/scikit-image/issues/1732).
        Also, the center must be given as (x,y), not (y,x), which is
        the opposite (!) of what the skimage documentation says...
        Such incomplete and contradictory documentation in skimage
        does not inspire much confidence (probably something will
        change in future versions), so instead we do the 180 deg
        rotation by hand. Also, since 180 deg is a very special kind
        of rotation, the current implementation is probably faster.

        """
        ny, nx = image.shape

        # Crop to region that can be rotated around center
        xc, yc = np.floor(center)
        dx = min(nx-1-xc, xc)
        dy = min(ny-1-yc, yc)
        xslice = slice(int(xc-dx), int(xc+dx+1))
        yslice = slice(int(yc-dy), int(yc+dy+1))
        image = image[yslice, xslice]
        image_180 = image[::-1, ::-1]

        # Redefine center
        center = np.array([dx, dy])

        # Note that aperture is defined for the new coordinates
        if kind == 'cas':
            r = self._petro_extent * self.petrosian_radius_circ
            ap = photutils.CircularAperture(center, r)
        elif kind == 'outer':
            r_in = self.half_light_radius
            r_out = self.rmax
            ap = photutils.CircularAnnulus(center, r_in, r_out)
        elif kind == 'shape':
            ap = photutils.CircularAperture(center, self.rmax)
        else:
            raise Exception('Asymmetry kind not understood:', kind)

        # Apply eq. 10 from Lotz et al. (2004)
        ap_abs_sum = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(np.abs(image_180-image), method='exact')[0][0]
        if kind == 'shape':
            # The shape asymmetry of the background is zero
            asym = ap_abs_diff / ap_abs_sum
        else:
            ap_area = _aperture_area(ap, self._mask_stamp)
            asym = (ap_abs_diff - ap_area*self._sky_asymmetry) / ap_abs_sum

        return asym

    @lazyproperty
    def _asymmetry_center(self):
        """
        Find the position of the central pixel (relative to the
        "postage stamp" cutout) that minimizes the (CAS) asymmetry.
        """
        image = self._cutout_stamp_maskzeroed

        # Initial guess
        center_0 = np.array([self._xc_stamp, self._yc_stamp])
        
        # Find minimum at pixel precision (xtol=1)
        center_asym = opt.fmin(self._asymmetry_function, center_0,
                               args=(image, 'cas'), xtol=1.0, disp=0)

        return np.floor(center_asym)

    @lazyproperty
    def asymmetry(self):
        """
        Calculate asymmetry as described in Lotz et al. (2004).
        """
        image = self._cutout_stamp_maskzeroed
        asym = self._asymmetry_function(self._asymmetry_center,
                                        image, 'cas')
        
        return asym

    @lazyproperty
    def concentration(self):
        """
        Calculate concentration as described in Lotz et al. (2004).
        """
        image = self._cutout_stamp_maskzeroed
        center = self._asymmetry_center
        r_max = self._petro_extent * self.petrosian_radius_circ
        
        r_20 = _radius_at_fraction_of_total(image, center, r_max, 0.2)
        r_80 = _radius_at_fraction_of_total(image, center, r_max, 0.8)
        
        return 5.0 * np.log10(r_80 / r_20)

    @lazyproperty
    def smoothness(self):
        """
        Calculate smoothness (a.k.a. clumpiness) as described in
        Lotz et al. (2004).
        """
        r = self._petro_extent * self.petrosian_radius_circ
        ap = photutils.CircularAperture(self._asymmetry_center, r)

        image = self._cutout_stamp_maskzeroed
        
        boxcar_size = int(self._petro_fraction_cas * self.petrosian_radius_circ)
        image_smooth = ndi.uniform_filter(image, size=boxcar_size)
        
        ap_abs_flux = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(
            np.abs(image_smooth - image), method='exact')[0][0]
        S = (ap_abs_diff - ap.area()*self._sky_smoothness) / ap_abs_flux

        return S

    ##################
    # MID statistics #
    ##################

    def _segmap_mid_main_clump(self, q):
        """
        For a given quantile `q`, return a boolean array indicating
        the locations of pixels above `q` that are also part of
        the "main" clump.
        """
        image = self._cutout_stamp_maskzeroed
        sorted_pixelvals = self._sorted_pixelvals_stamp

        threshold = _quantile(sorted_pixelvals, q)
        above_threshold = image >= threshold

        # Instead of assuming that the main segment is at the center
        # of the stamp, we use the already-calculated centroid:
        ic = int(self._yc_stamp)
        jc = int(self._xc_stamp)

        # Neighbor "footprint" for growing regions, including corners:
        s = ndi.generate_binary_structure(2, 2)

        labeled_array, num_features = ndi.label(above_threshold, structure=s)
        if labeled_array[ic, jc] == 0:
            # Centroid is not part of the main clump.
            return None

        return labeled_array == labeled_array[ic, jc]

    def _segmap_mid_function(self, q):
        """
        Helper function to calculate the MID segmap.
        
        For a given quantile `q`, return the ratio of the mean flux of
        pixels at the level of `q` (within the main clump) divided by
        the mean of pixels above `q` (within the main clump).
        """
        image = self._cutout_stamp_maskzeroed
        sorted_pixelvals = self._sorted_pixelvals_stamp

        locs_main_clump = self._segmap_mid_main_clump(q)
        mean_flux_main_clump = np.mean(image[locs_main_clump])
        mean_flux_new_pixels = _quantile(sorted_pixelvals, q)

        return mean_flux_new_pixels / mean_flux_main_clump - self._eta

    @lazyproperty
    def _segmap_mid_upper_bound(self):
        """
        Another helper function for the MID segmap. This one
        automatically finds an upper limit for the quantile that
        determines the MID segmap.
        """
        image = self._cutout_stamp_maskzeroed
        sorted_pixelvals = self._sorted_pixelvals_stamp

        num_bright_pixels = 2  # starting point
        while num_bright_pixels < image.size:
            q = 1.0 - float(num_bright_pixels) / float(image.size)
            if self._segmap_mid_main_clump(q) is None:
                num_bright_pixels = 2 * num_bright_pixels
            else:
                return q
        raise Exception('Should not reach this point.')

    @lazyproperty
    def _segmap_mid(self):
        """
        Create a new segmentation map as described in Section 4.3 from
        Freeman et al. (2013).
        
        Notes
        -----
        This implementation is independent of the number of quantiles
        used in the calculation, as well as other parameters.
        """
        image = self._cutout_stamp_maskzeroed
        sorted_pixelvals = self._sorted_pixelvals_stamp

        # Find appropriate quantile using numerical solver
        q_min = 0.0
        q_max = self._segmap_mid_upper_bound
        xtol = 1.0 / float(image.size)
        q = opt.brentq(self._segmap_mid_function, q_min, q_max, xtol=xtol)

        # Regularize a bit the shape of the segmap:
        locs_main_clump = self._segmap_mid_main_clump(q)
        segmap_float = ndi.uniform_filter(
            np.float64(locs_main_clump), size=self._boxcar_size_mid)
        segmap = segmap_float > 0.5

        return segmap

    @lazyproperty
    def _cutout_mid(self):
        """
        Apply the MID segmap to the postage stamp cutout
        and set negative pixels to zero.
        """
        image = np.where(self._segmap_mid,
                         self._cutout_stamp_maskzeroed, 0.0)
        image[image < 0] = 0.0
        return image

    @lazyproperty
    def _sorted_pixelvals_mid(self):
        """
        Just the sorted pixel values of the MID cutout.
        """
        return np.sort(self._cutout_mid.flatten())

    def _multimode_function(self, q):
        """
        Helper function to calculate the multimode statistic.
        Returns the sorted "areas" of the clumps at quantile `q`.
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
        invalid = self._cutout_mid.size  # high "energy" for basin-hopping
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
            elif mid_stepsize < 1.0 / self._cutout_mid.size:
                print('Warning: Single clump! (This should be rare.)')
                return 0.0
            else:
                mid_stepsize = mid_stepsize / 2.0
                print('Warning: Reduced stepsize to %g.' % (mid_stepsize))

        # STAGE 2: basin-hopping method

        # The results seem quite robust to changes in this parameter,
        # so I leave it hardcoded for now:
        mid_bh_rel_temp = 0.5

        temperature = -1.0 * mid_bh_rel_temp * ratio_min
        res = opt.basinhopping(self._multimode_ratio, q0,
            minimizer_kwargs={"method": "Nelder-Mead"},
            niter=self._niter_bh_mid, T=temperature, stepsize=mid_stepsize,
            interval=self._niter_bh_mid/2, disp=False)
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
        This is also much faster, thanks to the "peak_local_max" and
        "watershed" skimage routines.
        Returns a labeled array indicating regions around local maxima.
        
        """
        peaks = skimage.feature.peak_local_max(
            self._cutout_mid_smooth, indices=True, num_peaks=np.inf)
        num_peaks = peaks.shape[0]
        peak_labels = np.arange(num_peaks, dtype=np.int64)
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
        xp = sorted_xpeak[0] + 0.5  # center of pixel
        yp = sorted_ypeak[0] + 0.5
        
        # Calculate centroid
        m = skimage.measure.moments(image, order=1)
        yc = m[0, 1] / m[0, 0]
        xc = m[1, 0] / m[0, 0]

        area = np.sum(self._segmap_mid)

        return np.sqrt(np.pi/area) * np.sqrt((xp-xc)**2 + (yp-yc)**2)

    ###################
    # SHAPE ASYMMETRY #
    ###################

    @lazyproperty
    def half_light_radius(self):
        """
        The radius of a circular aperture containing 50% of the light,
        assuming that the center is at the brightest pixel and the total
        is at ``rmax`` (Pawlik et al. 2016).
        """
        image = self._cutout_stamp_maskzeroed
        
        # Center at brightest pixel
        center = np.array([self._x_maxval_stamp, self._y_maxval_stamp])
        
        return _radius_at_fraction_of_total(image, center, self.rmax, 0.5)

    @lazyproperty
    def radius_at_half_max(self):
        """
        The radius at half-maximum (half of the FWHM).
        """
        image = self._cutout_stamp_maskzeroed
        mask = self._mask_stamp
        r_max = self._dist_to_closest_corner
        r_half_max = _radius_at_fraction_of_maximum(
            image, mask, r_max, self._annulus_width, 0.5)
        
        return r_half_max

    @lazyproperty
    def _segmap_shape_asym(self):
        """
        Construct a binary detection mask as described in Section 3.1
        from Pawlik et al. (2016).
        
        Notes
        -----
        The algorithm seems to require a lot of sky area around the
        source of interest. In some cases we fall back to boundaries
        based on the Petrosian radius.
        
        """
        image = self._cutout_stamp_maskzeroed
        
        # Center at (center of) brightest pixel
        xc = self._x_maxval_stamp + 0.5
        yc = self._y_maxval_stamp + 0.5
        ic, jc = int(yc), int(xc)

        # Create a circular annulus around the brightest pixel
        # with inner and outer radii equal to 20 and 40 times
        # the *radius* at half-maximum (because 20 and 40 times
        # the FWHM seems like too much!).
        r_in = self._sky_num_fwhm * self.radius_at_half_max
        r_out = 2.0 * self._sky_num_fwhm * self.radius_at_half_max
        circ_annulus = photutils.CircularAnnulus((xc, yc), r_in, r_out)

        # Convert circular annulus aperture to binary mask
        circ_annulus_mask = circ_annulus.to_mask(method='center')[0]
        # With the same shape as ``image``
        circ_annulus_mask = circ_annulus_mask.to_image(image.shape)
        # Invert mask and exclude other sources
        total_mask = self._mask_stamp | np.logical_not(circ_annulus_mask)
        # Do sigma-clipping -- 5 iterations should be enough
        mean, median, std = sigma_clipped_stats(
            image, mask=total_mask, sigma=3.0, iters=5, cenfunc=_mode)

        # Mode as defined in Bertin & Arnouts (1996)
        mode = 2.5*median - 1.5*mean

        # Smooth image slightly and apply 1-sigma threshold
        image_smooth = ndi.uniform_filter(
            image, size=self._boxcar_size_shape_asym)
        threshold = mode + std
        above_threshold = image_smooth >= threshold

        # 8-connected neighbor "footprint" for growing regions:
        s = ndi.generate_binary_structure(2, 2)

        labeled_array, num_features = ndi.label(above_threshold, structure=s)
        if labeled_array[ic, jc] == 0:
            raise Exception('Brightest pixel is outside the main segment?')

        return labeled_array == labeled_array[ic, jc]

    @lazyproperty
    def rmax(self):
        """
        Return the distance (in pixels) from the brightest pixel
        to the edge of the main source segment, as defined in
        Pawlik et al. (2016).
        """
        image = self._cutout_stamp_maskzeroed
        ny, nx = image.shape

        # Center at (center of) brightest pixel
        xc = self._x_maxval_stamp + 0.5
        yc = self._y_maxval_stamp + 0.5

        # Distances from all pixels to the brightest pixel
        ypos, xpos = np.mgrid[0:ny, 0:nx] + 0.5  # center of pixel
        distances = np.sqrt((ypos-yc)**2 + (xpos-xc)**2)
        
        # Only consider pixels within the segmap.
        rmax = np.max(distances[self._segmap_shape_asym])

        return rmax

    @lazyproperty
    def _outer_asymmetry_center(self):
        """
        Find the position of the central pixel (relative to the
        "postage stamp" cutout) that minimizes the outer asymmetry.
        """
        image = self._cutout_stamp_maskzeroed
        
        # Initial guess
        center_0 = np.array([self._x_maxval_stamp, self._y_maxval_stamp])
        
        # Find minimum at pixel precision (xtol=1)
        center_asym = opt.fmin(self._asymmetry_function, center_0,
                               args=(image, 'outer'), xtol=1.0, disp=0)

        return np.floor(center_asym)

    @lazyproperty
    def outer_asymmetry(self):
        """
        Calculate outer asymmetry as described in Pawlik et al. (2016).
        """
        image = self._cutout_stamp_maskzeroed
        asym = self._asymmetry_function(self._outer_asymmetry_center,
                                        image, 'outer')
        
        return asym

    @lazyproperty
    def _shape_asymmetry_center(self):
        """
        Find the position of the central pixel (relative to the
        "postage stamp" cutout) that minimizes the shape asymmetry.
        """
        image = np.where(self._segmap_shape_asym, 1.0, 0.0)

        # Initial guess
        center_0 = np.array([self._x_maxval_stamp, self._y_maxval_stamp])
        
        # Find minimum at pixel precision (xtol=1)
        center_asym = opt.fmin(self._asymmetry_function, center_0,
                               args=(image, 'shape'), xtol=1.0, disp=0)

        return np.floor(center_asym)

    @lazyproperty
    def shape_asymmetry(self):
        """
        Calculate shape asymmetry as described in Pawlik et al. (2016).
        """
        image = np.where(self._segmap_shape_asym, 1.0, 0.0)
        asym = self._asymmetry_function(self._outer_asymmetry_center,
                                        image, 'shape')

        return asym


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

    References
    ----------
    Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

    """
    assert(image.shape == segmap.shape)
    if not isinstance(segmap, photutils.SegmentationImage):
        segmap = photutils.SegmentationImage(segmap)

    sources_morph = []
    for label in segmap.labels:
        sources_morph.append(SourceMorphology(image, segmap, label, **kwargs))

    return sources_morph

