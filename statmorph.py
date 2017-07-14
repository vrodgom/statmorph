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
from astropy.stats import gaussian_sigma_to_fwhm
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

class SourceMorphology(object):
    """
    Class to measure the morphological parameters of a single labeled
    source. The parameters can be accessed as attributes or keys.

    Parameters
    ----------
    image : array-like
        The 2D image containing the sources of interest.
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
        the size of the segment containing the source (the original
        implementation adds 100 pixels in each dimension). The value
        must be >= 1. The default value is 1.5 (i.e., 50% larger).
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
    sigma_mid : float, optional
        In the MID calculations, this is the smoothing scale (in pixels)
        used to compute the intensity (I) statistic. The default is 1.0.

    References
    ----------
    Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

    """
    def __init__(self, image, segmap, label, mask=None, variance=None,
                 cutout_extent=1.5, eta=0.2, petro_fraction_gini=0.2,
                 remove_outliers=False, n_sigma_outlier=10, border_size=5,
                 skybox_size=20, petro_extent=1.5, petro_fraction_cas=0.25,
                 boxcar_size_mid=3.0, sigma_mid=1.0):
        self._variance = variance
        self._cutout_extent = cutout_extent
        self._eta = eta
        self._petro_fraction_gini = petro_fraction_gini
        self._remove_outliers = remove_outliers
        self._n_sigma_outlier = n_sigma_outlier
        self._border_size = border_size
        self._skybox_size = skybox_size
        self._petro_extent = petro_extent
        self._petro_fraction_cas = petro_fraction_cas
        self._boxcar_size_mid = boxcar_size_mid
        self._sigma_mid = sigma_mid
        
        # The following object stores some important data:
        self._props = photutils.SourceProperties(image, segmap, label, mask=mask)

        # Centroid of the source relative to the "postage stamp" cutout:
        self._xc_stamp = self._props.xcentroid.value - self._slice_stamp[1].start
        self._yc_stamp = self._props.ycentroid.value - self._slice_stamp[0].start

    def __getitem__(self, key):
        return getattr(self, key)

    @lazyproperty
    def _slice_stamp(self):
        """
        Attempt to create a square slice (centered at the centroid)
        that is slightly larger than the minimum bounding box used by
        photutils. This is important for some morphological
        calculations. Note that the cutout may not be square when the
        source is close to a border of the original image.

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
    def _cutout_stamp_maskzeroed_double(self):
        """
        Return a data cutout centered on the source of interest,
        but which is slightly larger than the minimal bounding box.
        Pixels belonging to other sources (as well as masked pixels)
        are set to zero, but the background is left alone.

        """
        cutout_stamp = self._props._data[self._slice_stamp]
        segmap_stamp = self._props._segment_img.data[self._slice_stamp]
        mask_stamp = (segmap_stamp > 0) & (segmap_stamp != self._props.label)
        if self._props._mask is not None:
            mask_stamp = mask_stamp | self._props._mask[self._slice_stamp]
        cutout_stamp[mask_stamp] = 0
        
        # Some skimage functions require double precision:
        return np.float64(cutout_stamp)

    @lazyproperty
    def _sorted_pixelvals_stamp(self):
        """
        Just the sorted pixel values of the postage stamp.
        """
        return np.sort(self._cutout_stamp_maskzeroed_double.flatten())

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
        
        For the ellipse with semi-major axis `a`, return the
        ratio of the mean flux around the ellipse divided by
        the mean flux within the ellipse, minus "eta" (eq. 4
        from Lotz et al. 2004). The root of this function is
        the Petrosian "radius".

        """
        b = a / self._props.elongation.value
        a_in = a - 1.0
        a_out = a + 1.0
        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value

        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_stamp, self._yc_stamp), a_in, a_out, b_out, theta)
        ellip_aperture = photutils.EllipticalAperture(
            (self._xc_stamp, self._yc_stamp), a, b, theta)

        ellip_annulus_mean_flux = ellip_annulus.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0] / ellip_annulus.area()
        ellip_aperture_mean_flux = ellip_aperture.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0] / ellip_aperture.area()

        return ellip_annulus_mean_flux / ellip_aperture_mean_flux - self._eta

    def _petrosian_function_circ(self, r):
        """
        Helper function to calculate ``petrosian_radius_circ``.
        
        For the circle with radius `r`, return the
        ratio of the mean flux around the circle divided by
        the mean flux within the circle, minus "eta" (eq. 4
        from Lotz et al. 2004). The root of this function is
        the Petrosian radius.

        """
        r_in = r - 1.0
        r_out = r + 1.0

        circ_annulus = photutils.CircularAnnulus(
            (self._xc_stamp, self._yc_stamp), r_in, r_out)
        circ_aperture = photutils.CircularAperture(
            (self._xc_stamp, self._yc_stamp), r)

        circ_annulus_mean_flux = circ_annulus.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0] / circ_annulus.area()
        circ_aperture_mean_flux = circ_aperture.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0] / circ_aperture.area()

        return circ_annulus_mean_flux / circ_aperture_mean_flux - self._eta

    @lazyproperty
    def petrosian_radius_ellip(self):
        """
        Compute the Petrosian "radius" (actually the semi-major axis)
        for concentric elliptical apertures.
        
        Notes
        -----
        Instead of using a "curve of growth," we determine the Petrosian
        radius using a numerical solver. This should require less
        iterations, especially for large images.
        
        """
        a_min = 2.0
        a_max = self._dist_to_closest_corner
        rpetro_ellip = opt.brentq(self._petrosian_function_ellip,
                                  a_min, a_max, xtol=1e-6)
        
        return rpetro_ellip

    @lazyproperty
    def petrosian_radius_circ(self):
        """
        Compute the Petrosian radius for concentric circular apertures.
        
        Notes
        -----
        Instead of using a "curve of growth," we determine the Petrosian
        radius using a numerical solver. This should require less
        iterations, especially for large images.
        
        """
        r_min = 2.0
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
                self._cutout_stamp_maskzeroed_double, np.mean, footprint=local_footprint)
            local_std = ndi.filters.generic_filter(
                self._cutout_stamp_maskzeroed_double, np.std, footprint=local_footprint)
            bad_pixels = (np.abs(self._cutout_stamp_maskzeroed_double - local_mean) >
                          self._n_sigma_outlier * local_std)
            cutout_gini = np.where(~bad_pixels, self._cutout_stamp_maskzeroed_double, 0)
            
            print('There are %d bad pixels.' % (np.sum(bad_pixels)))
            print('It took', time.time() - start, 's to remove them.')

        else:
            cutout_gini = self._cutout_stamp_maskzeroed_double

        return cutout_gini

    @lazyproperty
    def _segmap_gini(self):
        """
        Create a new segmentation map (relative to the "Gini" cutout)
        based on the Petrosian "radius".
        
        Notes
        -----
        For simplicity, we tentatively remove the condition that the
        smoothing scale be at least 3 times the PSF scale, which is not
        mentioned in the original paper. Note that outliers have been
        removed before smoothing the image.
        
        """
        # Smooth image
        petro_sigma = self._petro_fraction_gini * self.petrosian_radius_ellip
        
        # TMP: Match IDL implementation (1/10 of fwhm)
        petro_sigma = petro_sigma * gaussian_sigma_to_fwhm / 2.0
        
        cutout_smooth = ndi.gaussian_filter(self._cutout_gini, petro_sigma)

        # Use mean flux at the Petrosian "radius" as threshold
        a_in = self.petrosian_radius_ellip - 1.0
        a_out = self.petrosian_radius_ellip + 1.0
        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value
        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_stamp, self._yc_stamp), a_in, a_out, b_out, theta)
        ellip_annulus_mean_flux = ellip_annulus.do_photometry(
            cutout_smooth, method='exact')[0][0] / ellip_annulus.area()
        segmap_gini = np.where(cutout_smooth >= ellip_annulus_mean_flux, 1, 0)
        
        return segmap_gini

    @lazyproperty
    def gini(self):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        
        """
        image = self._cutout_gini.flatten()
        segmap = self._segmap_gini.flatten()

        sorted_pixelvals = np.sort(np.abs(image[segmap == 1]))
        total_absflux = np.sum(sorted_pixelvals)

        n = len(sorted_pixelvals)
        indices = np.arange(1, n+1)  # start at i=1
        gini = np.sum((2*indices-n-1)*sorted_pixelvals) / (total_absflux*float(n-1))

        return gini

    @lazyproperty
    def m20(self):
        """
        Calculate the M_20 coefficient as described in Lotz et al. (2004).
        
        """
        # Use the same region as in the Gini calculation
        image = np.where(self._segmap_gini == 1, self._cutout_gini, 0.0)
        image = np.float64(image)  # skimage wants this

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
        locs = (self._segmap_gini > 0) & (image >= 0)
        pixelvals = image[locs]
        if self._variance is None:
            variance = np.zeros_like(pixelvals)
        else:
            variance = self._variance[self._slice_stamp][locs]

        return np.mean(image[locs] / np.sqrt(variance + self._sky_sigma**2))


    ##################
    # CAS statistics #
    ##################

    @lazyproperty
    def _slice_skybox(self):
        """
        Find a region of the sky that only contains background.
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
        return np.mean(self._cutout_stamp_maskzeroed_double[self._slice_skybox])

    @lazyproperty
    def _sky_sigma(self):
        """
        Mean background value.
        """
        return np.std(self._cutout_stamp_maskzeroed_double[self._slice_skybox])

    @lazyproperty
    def _sky_asymmetry(self):
        """
        Asymmetry of the background.
        """
        bkg = self._cutout_stamp_maskzeroed_double[self._slice_skybox]
        bkg_180 = bkg[::-1, ::-1]
        return np.sum(np.abs(bkg_180 - bkg)) / float(bkg.size)

    @lazyproperty
    def _sky_smoothness(self):
        """
        Smoothness of the background.
        """
        bkg = self._cutout_stamp_maskzeroed_double[self._slice_skybox]

        # If the smoothing "boxcar" is larger than the skybox itself,
        # this just sets all values equal to the mean:
        boxcar_size = int(self._petro_fraction_cas * self.petrosian_radius_circ)
        bkg_smooth = ndi.uniform_filter(bkg, size=boxcar_size)

        return np.sum(np.abs(bkg_smooth - bkg)) / float(bkg.size)

    def _asymmetry_function(self, center):
        """
        Helper function to determine the asymmetry and center of asymmetry.
        """
        r = self._petro_extent * self.petrosian_radius_circ
        ap = photutils.CircularAperture(center, r)

        image = self._cutout_stamp_maskzeroed_double

        # Here are some notes about why I'm *not* using
        # skimage.transform.rotate. The following line would
        # work correctly in skimage version 0.13.0:
        # skimage.transform.rotate(image, 180.0, center=np.floor(center))
        # However, note that the "center" argument must be truncated,
        # which is not mentioned in the documentation
        # (https://github.com/scikit-image/scikit-image/issues/1732).
        # Also, the center must be given as (x,y), not (y,x), which is
        # the opposite (!) of what the skimage documentation says...

        # Such incomplete and contradictory documentation in skimage
        # does not inspire much confidence (probably something will
        # change in future versions), so instead we do the 180 deg
        # rotation by hand. Also, this is probably faster:
        ny, nx = image.shape
        xc, yc = np.floor(center) + 0.5  # center of pixel
        dx = min(nx-xc, xc)
        dy = min(ny-yc, yc)
        # Crop to region that can be rotated around center:
        xslice = slice(int(xc-dx), int(xc+dx))
        yslice = slice(int(yc-dy), int(yc+dy))
        image = image[yslice, xslice]
        image_180 = image[::-1, ::-1]

        ap_abs_flux = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(np.abs(image_180-image), method='exact')[0][0]
        asym = (ap_abs_diff - ap.area()*self._sky_asymmetry) / ap_abs_flux

        return asym

    @lazyproperty
    def _asymmetry_center(self):
        """
        Find the position of the central pixel (relative to the
        "postage stamp" cutout) that minimizes the asymmetry.
        """
        # Initial guess
        center_0 = np.array([self._xc_stamp, self._yc_stamp])
        
        # Find minimum at pixel precision (xtol=1)
        center_asym = opt.fmin(self._asymmetry_function, center_0, xtol=1.0,
                               disp=0)

        return np.floor(center_asym)

    @lazyproperty
    def asymmetry(self):
        """
        Calculate asymmetry as described in Lotz et al. (2004).
        """
        return self._asymmetry_function(self._asymmetry_center)

    def _concentration_function(self, r, flux_fraction, flux_total):
        """
        Helper function to calculate the concentration.
        """
        ap = photutils.CircularAperture(self._asymmetry_center, r)
        ap_flux = ap.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0]

        return ap_flux / flux_total - flux_fraction

    @lazyproperty
    def concentration(self):
        """
        Calculate concentration as described in Lotz et al. (2004).
        """
        r_min = 2.0
        r_max = self._petro_extent * self.petrosian_radius_circ
        ap_total = photutils.CircularAperture(self._asymmetry_center, r_max)
        flux_total = ap_total.do_photometry(
            self._cutout_stamp_maskzeroed_double, method='exact')[0][0]
        
        r_20 = opt.brentq(self._concentration_function, r_min, r_max,
                          args=(0.2, flux_total), xtol=1e-6)
        r_80 = opt.brentq(self._concentration_function, r_min, r_max,
                          args=(0.8, flux_total), xtol=1e-6)
        
        return 5.0 * np.log10(r_80 / r_20)

    @lazyproperty
    def smoothness(self):
        """
        Calculate smoothness (a.k.a. clumpiness) as described in
        Lotz et al. (2004).
        """
        r = self._petro_extent * self.petrosian_radius_circ
        ap = photutils.CircularAperture(self._asymmetry_center, r)

        image = self._cutout_stamp_maskzeroed_double
        
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
        image = self._cutout_stamp_maskzeroed_double
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
        image = self._cutout_stamp_maskzeroed_double
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
        image = self._cutout_stamp_maskzeroed_double
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
        image = self._cutout_stamp_maskzeroed_double
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
                        self._cutout_stamp_maskzeroed_double, 0.0)
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
        # quantiles, equal to twice the value used in the original IDL
        # implementation. If every calculated ratio is invalid, we
        # try a smaller size.
        mid_stepsize = 0.04

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

        # The results seem quite robust to changes in the
        # following two parameters, so I just leave them
        # hardcoded here for now:
        mid_bh_niter = 5
        mid_bh_rel_temp = 0.2

        temperature = -1.0 * ratio_min * mid_bh_rel_temp
        res = opt.basinhopping(self._multimode_ratio, q0,
            minimizer_kwargs={"method": "Nelder-Mead"},
            niter=mid_bh_niter, T=temperature, stepsize=mid_stepsize,
            interval=mid_bh_niter, disp=False)
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
        sorted_flux_sums, sorted_xpeak, sorted_ypeak = self._intensity_sums
        xp = sorted_xpeak[0] + 0.5  # center of pixel
        yp = sorted_ypeak[0] + 0.5
        
        # Calculate centroid
        m = skimage.measure.moments(self._cutout_mid, order=1)
        yc = m[0, 1] / m[0, 0]
        xc = m[1, 0] / m[0, 0]

        area = np.sum(self._segmap_mid)

        return np.sqrt(np.pi/area) * np.sqrt((xp-xc)**2 + (yp-yc)**2)


def source_morphology(image, segmap, **kwargs):
    """
    Calculate the morphological parameters of all sources in ``image``
    as labeled by ``segmap``.
    
    Parameters
    ----------
    image : array-like
        The 2D image containing the sources of interest.
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

