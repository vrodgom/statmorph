import numpy as np
import time
import scipy.optimize as opt
import scipy.ndimage as ndi
import skimage.measure as msr
from skimage.transform import rotate
from astropy.utils import lazyproperty
from astropy.stats import gaussian_sigma_to_fwhm
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
        The fraction of the Petrosian radius used as a smoothing scale
        in order to define the pixels that belong to the galaxy, at
        least for the calculation of the Gini coefficient. The default
        value is 0.2.
    remove_outliers : bool, optional
        If `True`, remove outlying pixels as described in Lotz et al.
        (2004), using the parameter ``n_sigma_outlier``. This is the
        most time-consuming operation and, at least for reasonably
        clean data, it should have a negligible effect on Gini
        coefficient. By default it is set to `False`.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a pixel as an
        outlier, relative to its 8 neighbors. This parameter only
        takes effect when ``remove_outliers`` is `True`. The default
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

    References
    ----------
    Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

    """
    def __init__(self, image, segmap, label, mask=None, cutout_extent=1.5,
                 eta=0.2, petro_sigma_fraction=0.2, remove_outliers=False,
                 n_sigma_outlier=10, border_size=5, skybox_size=20,
                 petro_extent=1.5):
        self._cutout_extent = cutout_extent
        self._eta = eta
        self._petro_sigma_fraction = petro_sigma_fraction
        self._remove_outliers = remove_outliers
        self._n_sigma_outlier = n_sigma_outlier
        self._border_size = border_size
        self._skybox_size = skybox_size
        self._petro_extent = petro_extent
        
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
        petro_sigma = self._petro_sigma_fraction * self.petrosian_radius_ellip
        
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
        m = msr.moments(image, order=1)
        yc = m[0, 1] / m[0, 0]
        xc = m[1, 0] / m[0, 0]

        # Calculate second total central moment
        mc = msr.moments_central(image, yc, xc, order=3)
        second_moment = mc[0, 2] + mc[2, 0]

        # Calculate threshold pixel value
        sorted_pixelvals = np.sort(image.flatten())
        flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
        threshold = sorted_pixelvals[flux_fraction >= 0.8][0]

        # Calculate second moment of the brightest pixels
        image_20 = np.where(image >= threshold, image, 0.0)
        mc_20 = msr.moments_central(image_20, yc, xc, order=3)
        second_moment_20 = mc_20[0, 2] + mc_20[2, 0]

        # Normalize
        m20 = np.log10(second_moment_20 / second_moment)

        return m20

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
        bkg_180 = rotate(bkg, 180.0)

        #~ # This gives the same result:
        #~ bkg_180 = bkg[::-1, ::-1]

        return np.sum(np.abs(bkg_180 - bkg)) / float(bkg.size)

    def _asymmetry_function(self, center):
        """
        Helper function to determine the asymmetry and center of asymmetry.
        
        """
        r = self._petro_extent * self.petrosian_radius_circ
        ap = photutils.CircularAperture(center, r)

        image = self._cutout_stamp_maskzeroed_double
        # Some comments about skimage version 0.13.0...
        # The "center" argument must be truncated
        # (https://github.com/scikit-image/scikit-image/issues/1732).
        # Also, the center must be given as (x,y), not (y,x), which is
        # the opposite of what the skimage documentation says...
        image_180 = rotate(image, 180.0, center=np.floor(center))

        #~ # Alternatively, this is a less precise (but more trustworthy
        #~ # and probably faster) way of doing the 180 deg rotation:
        #~ xc, yc = np.floor(center) + 0.5  # center of pixel
        #~ ny, nx = image.shape
        #~ dx = min(nx-xc, xc)
        #~ dy = min(ny-yc, yc)
        #~ xslice = slice(int(xc-dx), int(xc+dx))
        #~ yslice = slice(int(yc-dy), int(yc+dy))
        #~ # Limit to region that can be rotated around center:
        #~ image = image[yslice, xslice]
        #~ image_180 = image[::-1, ::-1]

        ap_abs_flux = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(np.abs(image_180-image), method='exact')[0][0]
        asym = (ap_abs_diff - ap.area()*self._sky_asymmetry) / ap_abs_flux

        return asym

    @lazyproperty
    def asymmetry(self):
        # Preliminary calculation
        
        center_0 = np.array([self._xc_stamp, self._yc_stamp])
        
        return self._asymmetry_function(center_0)
        

def source_morphology(image, segmap, mask=None, cutout_extent=1.5,
                 eta=0.2, petro_sigma_fraction=0.2, remove_outliers=False,
                 n_sigma_outlier=10, border_size=5, skybox_size=20,
                 petro_extent=1.5):
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
        The fraction of the Petrosian radius used as a smoothing scale
        in order to define the pixels that belong to the galaxy, at
        least for the calculation of the Gini coefficient. The default
        value is 0.2.
    remove_outliers : bool, optional
        If `True`, remove outlying pixels as described in Lotz et al.
        (2004), using the parameter ``n_sigma_outlier``. This is the
        most time-consuming operation and, at least for reasonably
        clean data, it should have a negligible effect on Gini
        coefficient. By default it is set to `False`.
    n_sigma_outlier : scalar, optional
        The number of standard deviations that define a pixel as an
        outlier, relative to its 8 neighbors. This parameter only
        takes effect when ``remove_outliers`` is `True`. The default
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
        sources_morph.append(SourceMorphology(
            image, segmap, label, mask=mask, cutout_extent=cutout_extent,
            eta=eta, petro_sigma_fraction=petro_sigma_fraction,
            remove_outliers=remove_outliers, n_sigma_outlier=n_sigma_outlier,
            border_size=border_size, skybox_size=skybox_size,
            petro_extent=petro_extent))

    return sources_morph



