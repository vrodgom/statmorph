import numpy as np
import time
import scipy.optimize as opt
import scipy.ndimage as ndi
from astropy.utils import lazyproperty
import photutils

__all__ = ['SourceMorphology', 'source_morphology']

class SourceMorphology(object):
    """
    Class to measure the morphological parameters of a single labeled
    source. The parameters can be accessed as attributes or keys.

    Parameters
    ----------
    image : array-like
        The 2D image containing the sources of interest.
    segmap : array-like (int) or `photutils.SegmentationImage`
        A 2D segmentation map where different sources are 
        indicated with different positive integer values.
        A value of zero represents the background.
    label : int
        A label indicating the source of interest.
    mask : array-like (bool), optional
        A 2D array with the same size as ``image``, where pixels
        set to `True` are ignored from all calculations.
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
    petro_sigma_fraction : float, optional
        The fraction of the Petrosian radius used as
        a smoothing scale before defining the pixels
        that belong to the galaxy, at least for the
        calculation of the Gini coefficient. The
        default value is 0.2.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a
        pixel as an outlier, relative to its 8 neighbors.
        The default value is 10.

    References
    ----------
    Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

    """
    def __init__(self, image, segmap, label, mask=None, cutout_extent=1.5,
                 eta=0.2, petro_sigma_fraction=0.2, n_sigma_outlier=10):
        self._cutout_extent = cutout_extent
        self._eta = eta
        self._petro_sigma_fraction = petro_sigma_fraction
        self._n_sigma_outlier = n_sigma_outlier

        # The following object stores some important data:
        self._props = photutils.SourceProperties(image, segmap, label, mask=mask)

        # Centroid of the source relative to the "morphology" cutout:
        self._xc_morph = self._props.xcentroid.value - self._slice_morph[1].start
        self._yc_morph = self._props.ycentroid.value - self._slice_morph[0].start

    def __getitem__(self, key):
        return getattr(self, key)

    @lazyproperty
    def _slice_morph(self):
        """
        Attempt to create a square "slice" (centered at the centroid)
        that is slightly larger than the minimum bounding box used by
        photutils. This is necessary for some morphological
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
        slice_square = (slice(max(0, yc-dist), min(ny, yc+dist)),
                        slice(max(0, xc-dist), min(nx, xc+dist)))

        return slice_square

    @lazyproperty
    def _cutout_morph(self):
        """
        Return a data cutout centered on the source of interest,
        but which is slightly larger than the minimal bounding box.
        Pixels belonging to other sources (as well as masked pixels)
        are set to zero, but the background is left alone.
        
        """
        cutout_morph = self._props._data[self._slice_morph]
        segmap_morph = self._props._segment_img.data[self._slice_morph]

        cutout_morph = np.where(
            (segmap_morph == 0) | (segmap_morph == self._props.label),
            cutout_morph, 0)

        if self._props._mask is not None:
            cutout_morph = np.where(~self._props._mask, cutout_morph, 0)
        
        return cutout_morph

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
            (self._xc_morph, self._yc_morph), a_in, a_out, b_out, theta)
        ellip_aperture = photutils.EllipticalAperture(
            (self._xc_morph, self._yc_morph), a, b, theta)

        ellip_annulus_mean_flux = ellip_annulus.do_photometry(
            self._cutout_morph, method='exact')[0][0] / ellip_annulus.area()
        ellip_aperture_mean_flux = ellip_aperture.do_photometry(
            self._cutout_morph, method='exact')[0][0] / ellip_aperture.area()

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
            (self._xc_morph, self._yc_morph), r_in, r_out)
        circ_aperture = photutils.CircularAperture(
            (self._xc_morph, self._yc_morph), r)

        circ_annulus_mean_flux = circ_annulus.do_photometry(
            self._cutout_morph, method='exact')[0][0] / circ_annulus.area()
        circ_aperture_mean_flux = circ_aperture.do_photometry(
            self._cutout_morph, method='exact')[0][0] / circ_aperture.area()

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

    @lazyproperty
    def _cutout_clean(self):
        """
        Remove outliers from the cutout containing the source.
        
        """
        local_footprint = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        local_mean = ndi.filters.generic_filter(
            self._cutout, np.mean, footprint=local_footprint)
        local_sigma = ndi.filters.generic_filter(
            self._cutout, np.std, footprint=local_footprint)
        cutout_clean = np.where(
            self._cutout < local_mean + self._n_sigma_outlier * local_sigma,
            self._cutout, 0)

        return cutout_clean

    @lazyproperty
    def _cutout_gini(self):
        """
        Remove outliers as described in Lotz et al. (2004).
        """
        local_footprint = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        local_mean = ndi.filters.generic_filter(
            self._cutout_morph, np.mean, footprint=local_footprint)
        local_std = ndi.filters.generic_filter(
            self._cutout_morph, np.std, footprint=local_footprint)
        cutout_gini = np.where(
            np.abs(self._cutout_morph - local_mean) < self._n_sigma_outlier * local_std,
            self._cutout_morph, 0)
        
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
        petro_sigma = self._petro_sigma_fraction * self.petrosian_radius_ellip

        # Smooth image using ndimage (note: scipy.fftconvolve and
        # astropy.convolve_fft take exactly the same time)
        start = time.time()
        cutout_smooth = ndi.gaussian_filter(self._cutout_gini, petro_sigma)

        #~ # Smooth image -- try Astropy
        #~ import astropy.convolution as conv
        #~ gauss = conv.Gaussian2DKernel(petro_sigma)
        #~ cutout_smooth = conv.convolve_fft(self._cutout_gini, gauss)

        #~ # Smooth image -- try scipy.signal.fftconvolve
        #~ import scipy.signal as sig
        #~ # 4 standard deviations on each side seems quite standard
        #~ kernel = np.outer(sig.gaussian(int(8*petro_sigma), petro_sigma), 
                          #~ sig.gaussian(int(8*petro_sigma), petro_sigma))
        #~ cutout_smooth = sig.fftconvolve(self._cutout_gini, kernel, mode='same')

        print('Time spent smoothing image:', time.time() - start, 's.')
        
        # Use mean flux at the Petrosian "radius" as threshold
        a_in = self.petrosian_radius_ellip - 1.0
        a_out = self.petrosian_radius_ellip + 1.0
        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value
        ellip_annulus = photutils.EllipticalAnnulus(
            (self._xc_morph, self._yc_morph), a_in, a_out, b_out, theta)
        ellip_annulus_mean_flux = ellip_annulus.do_photometry(
            cutout_smooth, method='exact')[0][0] / ellip_annulus.area()
        petro_segmap = np.where(cutout_smooth >= ellip_annulus_mean_flux, 1, 0)
        
        return petro_segmap

    @lazyproperty
    def gini(self):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        
        Notes
        -----
        The Gini calculation is ambiguous when the image has negative values
        (e.g., from sky subtraction). Although this function considers the
        absolute value of the pixels, the negative values in the data have
        already been set to zero.
        
        """
        image = self._cutout_gini.flatten()
        segmap = self._segmap_gini.flatten()

        sorted_pixelvals = np.sort(np.abs(image[segmap == 1]))
        total_absflux = np.sum(sorted_pixelvals)

        # Fast computation using array operations
        n = len(sorted_pixelvals)
        indices = np.arange(1, n+1, dtype=np.int64)  # start at i=1
        gini = np.sum((2*indices-n-1)*sorted_pixelvals) / (total_absflux*float(n-1))

        return gini


def source_morphology(image, segmap):
    """
    Calculate the morphological parameters of all sources in ``image``
    as defined by ``segmap``.
    
    Parameters
    ----------
    image : array-like
        The 2D image containing the sources of interest.
    segmap : array-like (int) or `photutils.SegmentationImage`
        A 2D segmentation map where different sources are 
        indicated with different positive integer values.
        A value of zero represents the background.
    mask : array-like (bool), optional
        A 2D array with the same size as ``image``, where pixels
        set to `True` are ignored from all calculations.
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
    petro_sigma_fraction : float, optional
        The fraction of the Petrosian radius used as
        a smoothing scale before defining the pixels
        that belong to the galaxy, at least for the
        calculation of the Gini coefficient. The
        default value is 0.2.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a
        pixel as an outlier, relative to its 8 neighbors.
        The default value is 10.

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
        sources_morph.append(SourceMorphology(image, segmap, label))

    return sources_morph



