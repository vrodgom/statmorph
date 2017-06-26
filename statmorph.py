import numpy as np
import scipy.optimize as opt
from astropy.utils import lazyproperty
import photutils

__all__ = ['SourceMorphology', 'source_morphology']

class SourceMorphology(object):
    """
    Class to measure morphological parameters of a single source.
    The morphological parameters can be accessed as attributes or keys.

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
    eta : float, optional
          The Petrosian ``eta`` parameter used to define the Petrosian
          radius. For a circular or elliptical aperture at the Petrosian
          radius, the mean flux at the edge of the aperture divided by
          the mean flux within the aperture is equal to ``eta``. The
          default value is typically set to 0.2 (Petrosian 1976).

    References
    ----------
    Lotz J. M., Primack J., Madau P., 2004, AJ, 128, 163

    """
    def __init__(self, image, segmap, label, mask=None, eta=0.2):
        self.eta = eta
        self._props = photutils.SourceProperties(image, segmap, label, mask=mask)

    def __getitem__(self, key):
        return getattr(self, key)

    @lazyproperty
    def _dist_to_closest_corner(self):
        """
        The distance from the centroid to the closest corner of the
        bounding box containing the source segment. This is used as an
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
        a_in = a - 1
        a_out = a + 1
        b_out = a_out / self._props.elongation.value
        theta = self._props.orientation.value

        # Centroid of the source relative to the cutout:
        xc = self._props.xcentroid.value - self._props.xmin.value
        yc = self._props.ycentroid.value - self._props.ymin.value

        ellip_annulus = photutils.EllipticalAnnulus(
            (xc, yc), a_in, a_out, b_out, theta)
        ellip_aperture = photutils.EllipticalAperture(
            (xc, yc), a, b, theta)

        ellip_annulus_mean_flux = (ellip_annulus.do_photometry(
            self._props._data_cutout_maskzeroed_double, method='exact')[0][0] /
            ellip_annulus.area())
        ellip_aperture_mean_flux = (ellip_aperture.do_photometry(
            self._props._data_cutout_maskzeroed_double, method='exact')[0][0] /
            ellip_aperture.area())

        return ellip_annulus_mean_flux / ellip_aperture_mean_flux - self.eta

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

        # Determine Petrosian radius to a millionth of a pixel:
        rpetro_ellip = opt.brentq(self._petrosian_function_ellip,
                                  a_min, a_max, xtol=1e-6)
        
        return rpetro_ellip

    def _petrosian_function_circ(self, r):
        """
        Helper function to calculate ``petrosian_radius_circ``.
        
        For the circle with radius `r`, return the
        ratio of the mean flux around the circle divided by
        the mean flux within the circle, minus "eta" (eq. 4
        from Lotz et al. 2004). The root of this function is
        the Petrosian radius.

        """
        r_in = r - 1
        r_out = r + 1

        # Centroid of the source relative to the cutout:
        xc = self._props.xcentroid.value - self._props.xmin.value
        yc = self._props.ycentroid.value - self._props.ymin.value

        circ_annulus = photutils.CircularAnnulus(
            (xc, yc), r_in, r_out)
        circ_aperture = photutils.CircularAperture(
            (xc, yc), r)

        circ_annulus_mean_flux = (circ_annulus.do_photometry(
            self._props._data_cutout_maskzeroed_double, method='exact')[0][0] /
            circ_annulus.area())
        circ_aperture_mean_flux = (circ_aperture.do_photometry(
            self._props._data_cutout_maskzeroed_double, method='exact')[0][0] /
            circ_aperture.area())

        return circ_annulus_mean_flux / circ_aperture_mean_flux - self.eta

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

        # Determine Petrosian radius to a millionth of a pixel:
        rpetro_circ = opt.brentq(self._petrosian_function_circ,
                                 r_min, r_max, xtol=1e-6)
        
        return rpetro_circ


def source_morphology(image, segmap):
    """
    Calculate morphological parameters of all sources defined by segmap.
    
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
    eta : float, optional
          The Petrosian ``eta`` parameter used to define the Petrosian
          radius. For a circular or elliptical aperture at the Petrosian
          radius, the mean flux at the edge of the aperture divided by
          the mean flux within the aperture is equal to ``eta``. The
          default value is typically set to 0.2 (Petrosian 1976).

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


def compute_gini(image, segmap):
    """
    Calculate the Gini coefficient as in eq. (3) from Lotz et al. (2004).
    
    Parameters
    ----------
    image : array_like
            Image to be analyzed.
    segmap : array_like
            Segmentation map with the same shape as "image". The main
            galaxy segment should be indicated with 1; other values are 0.

    Returns
    -------
    gini : float
           The computed Gini value.

    Notes
    -----
    The Gini calculation is ambiguous when the image has negative values
    (e.g., from sky subtraction). In these cases we consider the
    absolute value of negative pixels, even if this means that the
    ordering of pixels with small values is not perfectly conserved.
    
    """
    image = np.asarray(image).flatten()
    segmap = np.asarray(segmap).flatten()

    sorted_pixelvals = np.sort(np.abs(image[segmap == 1]))
    total_absflux = np.sum(sorted_pixelvals)

    # Fast computation using numpy arrays
    n = len(sorted_pixelvals)
    indices = np.arange(1, n+1, dtype=np.int64)  # start at i=1
    gini = np.sum((2*indices-n-1)*sorted_pixelvals) / (total_absflux*float(n-1))

    return gini




