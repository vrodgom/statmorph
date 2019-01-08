"""
This file defines the `make_figure` function, which can be useful for
debugging and/or examining the morphology of a source in detail.
"""
# Author: Vicente Rodriguez-Gomez <v.rodriguez@irya.unam.mx>
# Licensed under a 3-Clause BSD License.

import numpy as np
import warnings
import time
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    if sys.version_info[0] == 2:  # Python 2
        matplotlib.use('agg')
    elif sys.version_info[0] == 3:  # Python 3
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import scipy.signal
import scipy.ndimage as ndi
import skimage.transform
from astropy.io import fits
from astropy.visualization import LogStretch

__all__ = ['make_figure']

def normalize(image, m=None, M=None):
    if m is None:
        m = np.min(image)
    if M is None:
        M = np.max(image)

    retval = (image-m) / (M-m)
    retval[image <= m] = 0.0
    retval[image >= M] = 1.0

    return retval

def get_ax(fig, row, col, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig):
    x_ax = (col+1)*eps + col*wpanel
    y_ax = eps + (nrows-1-row)*(hpanel+htop)
    return fig.add_axes([x_ax/wfig, y_ax/hfig, wpanel/wfig, hpanel/hfig])

def make_figure(morph):
    """
    Creates a figure analogous to Fig. 4 from Rodriguez-Gomez et al. (2019)
    for a given ``SourceMorphology`` object.
    
    Parameters
    ----------
    morph : ``statmorph.SourceMorphology``
        An object containing the morphological measurements of a single
        source.

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        The figure.

    """
    # I'm tired of dealing with plt.add_subplot, plt.subplots, plg.GridSpec,
    # plt.subplot2grid, etc. and never getting the vertical and horizontal
    # inter-panel spacings to have the same size, so instead let's do
    # everything manually:
    nrows = 2
    ncols = 4
    wpanel = 4.0  # panel width
    hpanel = 4.0  # panel height
    htop = 0.05*nrows*hpanel  # top margin and vertical space between panels
    eps = 0.005*nrows*hpanel  # all other margins
    wfig = ncols*wpanel + (ncols+1)*eps  # total figure width
    hfig = nrows*(hpanel+htop) + eps  # total figure height
    fig = plt.figure(figsize=(wfig, hfig))

    # For drawing circles/ellipses
    theta_vec = np.linspace(0.0, 2.0*np.pi, 200)

    # Add black to pastel colormap
    cmap_orig = matplotlib.cm.Pastel1
    colors = ((0.0, 0.0, 0.0), *cmap_orig.colors)
    cmap = matplotlib.colors.ListedColormap(colors)

    log_stretch = LogStretch(a=10000.0)

    # Get some general info about the image
    image = np.float64(morph._cutout_stamp_maskzeroed)  # skimage wants double
    ny, nx = image.shape
    m = np.min(image)
    M = np.max(image)
    m_stretch, M_stretch = log_stretch([m, M])
    xc, yc = morph._xc_stamp, morph._yc_stamp  # centroid
    xca, yca = morph._asymmetry_center  # asym. center
    xcs, ycs = morph._sersic_model.x_0.value, morph._sersic_model.y_0.value  # Sersic center

    # Plot everything w.r.t. centers of pixels (instead of lower-left corners):
    xc += 0.5; yc += 0.5; xca += 0.5; yca += 0.5; xcs += 0.5; ycs += 0.5

    ##################
    # Original image #
    ##################
    ax = get_ax(fig, 0, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)), cmap='gray', origin='lower',
              vmin=m_stretch, vmax=M_stretch)

    ax.plot(xc, yc, 'go', markersize=5, label='Centroid')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_centroid
    x0, x1 = xc - R*np.cos(theta), xc + R*np.cos(theta)
    y0, y1 = yc - R*np.sin(theta), yc + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'g--', lw=1.5, label='Major Axis (Centroid)')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_asymmetry
    x0, x1 = xca - R*np.cos(theta), xca + R*np.cos(theta)
    y0, y1 = yca - R*np.sin(theta), yca + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'b--', lw=1.5, label='Major Axis (Asym.)')
    # Half-radius ellipse
    a = morph.rhalf_ellip
    b = a / morph.elongation_asymmetry
    theta = morph.orientation_asymmetry
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'b', label='Half-Light Ellipse')
    # Some text
    text = 'flag = %d\nEllip. (Centroid) = %.4f\nEllip. (Asym.) = %.4f' % (
        morph.flag, morph.ellipticity_centroid, morph.ellipticity_asymmetry)
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Original Image (Log Stretch)', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##############
    # Sersic fit #
    ##############
    ax = get_ax(fig, 0, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = morph._sersic_model(x, y)
    # Add background noise (for realism)
    if morph.sky_sigma > 0:
        sersic_model += np.random.normal(scale=morph.sky_sigma, size=(ny, nx))
    ax.imshow(log_stretch(normalize(sersic_model, m=m, M=M)), cmap='gray',
              origin='lower', vmin=m_stretch, vmax=M_stretch)
    ax.plot(xcs, ycs, 'ro', markersize=5, label='Sérsic Center')
    R = float(nx**2 + ny**2)
    theta = morph.sersic_theta
    x0, x1 = xcs - R*np.cos(theta), xcs + R*np.cos(theta)
    y0, y1 = ycs - R*np.sin(theta), ycs + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'r--', lw=1.5, label='Major Axis (Sérsic)')
    # Half-radius ellipse
    a = morph.sersic_rhalf
    b = a * (1.0 - morph.sersic_ellip)
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xcs + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = ycs + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'r', label='Half-Light Ellipse (Sérsic)')
    # Some text
    text = ('flag_sersic = %d' % (morph.flag_sersic) + '\n' +
            'Ellip. (Sérsic) = %.4f' % (morph.sersic_ellip) + '\n' +
            r'$n = %.4f$' % (morph.sersic_n))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Sérsic Model + Noise', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Sersic residual #
    ###################
    ax = get_ax(fig, 0, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_res = morph._cutout_stamp_maskzeroed - morph._sersic_model(x, y)
    sersic_res[morph._mask_stamp] = 0.0
    ax.imshow(normalize(sersic_res), cmap='gray', origin='lower')
    ax.set_title('Sérsic Residual, ' + r'$I - I_{\rm model}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ######################
    # Asymmetry residual #
    ######################
    ax = get_ax(fig, 0, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    # Rotate image around asym. center
    # (note that skimage expects pixel positions at lower-left corners)
    image_180 = skimage.transform.rotate(image, 180.0, center=(xca-0.5, yca-0.5))
    image_res = image - image_180
    # Apply symmetric mask
    mask = morph._mask_stamp.copy()
    mask_180 = skimage.transform.rotate(mask, 180.0, center=(xca-0.5, yca-0.5))
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image_res = np.where(~mask_symmetric, image_res, 0.0)
    ax.imshow(normalize(image_res), cmap='gray', origin='lower')
    ax.set_title('Asymmetry Residual, ' + r'$I - I_{180}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Original segmap #
    ###################
    ax = get_ax(fig, 1, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)), cmap='gray', origin='lower',
              vmin=m_stretch, vmax=M_stretch)
    # Show original segmap
    contour_levels = [0.5]
    contour_colors = [(0,0,0)]
    segmap_stamp = morph._segmap.data[morph._slice_stamp]
    Z = np.float64(segmap_stamp == morph.label)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Show skybox
    xmin = morph._slice_skybox[1].start
    ymin = morph._slice_skybox[0].start
    xmax = morph._slice_skybox[1].stop - 1
    ymax = morph._slice_skybox[0].stop - 1
    ax.plot(np.array([xmin, xmax, xmax, xmin, xmin]) + 0.5,
            np.array([ymin, ymin, ymax, ymax, ymin]) + 0.5,
            'b', lw=1.5, label='Skybox')
    # Some text
    text = ('Sky Mean = %.4f' % (morph.sky_mean) + '\n' +
            'Sky Median = %.4f' % (morph.sky_median) + '\n' +
            'Sky Sigma = %.4f' % (morph.sky_sigma))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Original Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###############
    # Gini segmap #
    ###############
    ax = get_ax(fig, 1, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)),
              cmap='gray', origin='lower', vmin=m_stretch, vmax=M_stretch)
    # Show Gini segmap
    contour_levels = [0.5]
    contour_colors = [(0,0,0)]
    Z = np.float64(morph._segmap_gini)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Some text
    text = r'$\left\langle {\rm S/N} \right\rangle = %.4f$' % (morph.sn_per_pixel)
    ax.text(0.034, 0.966, text, fontsize=12,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$G = %.4f$' % (morph.gini) + '\n' +
            r'$M_{20} = %.4f$' % (morph.m20) + '\n' +
            r'$F(G, M_{20}) = %.4f$' % (morph.gini_m20_bulge) + '\n' +
            r'$S(G, M_{20}) = %.4f$' % (morph.gini_m20_merger))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$C = %.4f$' % (morph.concentration) + '\n' +
            r'$A = %.4f$' % (morph.asymmetry) + '\n' +
            r'$S = %.4f$' % (morph.smoothness))
    ax.text(0.966, 0.034, text, fontsize=12,
        horizontalalignment='right', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Gini Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ####################
    # Watershed segmap #
    ####################
    ax = get_ax(fig, 1, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    labeled_array, peak_labels, xpeak, ypeak = morph._watershed_mid
    labeled_array_plot = (labeled_array % (cmap.N-1)) + 1
    labeled_array_plot[labeled_array == 0] = 0.0  # background is black
    ax.imshow(labeled_array_plot, cmap=cmap, origin='lower',
              norm=matplotlib.colors.NoNorm())
    sorted_flux_sums, sorted_xpeak, sorted_ypeak = morph._intensity_sums
    if len(sorted_flux_sums) > 0:
        ax.plot(sorted_xpeak[0] + 0.5, sorted_ypeak[0] + 0.5, 'bo', markersize=2,
                label='First Peak')
    if len(sorted_flux_sums) > 1:
        ax.plot(sorted_xpeak[1] + 0.5, sorted_ypeak[1] + 0.5, 'ro', markersize=2,
                label='Second Peak')
    # Some text
    text = (r'$M = %.4f$' % (morph.multimode) + '\n' +
            r'$I = %.4f$' % (morph.intensity) + '\n' +
            r'$D = %.4f$' % (morph.deviation))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Watershed Segmap (' + r'$I$' + ' statistic)', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##########################
    # Shape asymmetry segmap #
    ##########################
    ax = get_ax(fig, 1, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(morph._segmap_shape_asym, cmap='gray', origin='lower')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    r = morph.rpetro_circ
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'b',
            label=r'$r_{\rm petro, circ}$')
    r = morph.rpetro_ellip
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r',
            label=r'$r_{\rm petro, ellip}$')
    r = morph.rmax_circ
    ax.plot(np.floor(xca) + r*np.cos(theta_vec), np.floor(yca) + r*np.sin(theta_vec),
            'c', lw=1.5, label=r'$r_{\rm max}$')
    text = (r'$A_S = %.4f$' % (morph.shape_asymmetry))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Shape Asymmetry Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.subplots_adjust(left=eps/wfig, right=1-eps/wfig, bottom=eps/hfig,
                        top=1.0-htop/hfig, wspace=eps/wfig, hspace=htop/hfig)

    #fig.savefig('test_segmap.png', dpi=150)
    
    return fig
