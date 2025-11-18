import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')
from matplotlib.gridspec import GridSpec

from pathlib import Path

import astropy
from astropy.table import Table, hstack, unique
from astropy.table import join

import os

plt.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
# Create custom colormaps for plots
pink_blue_2val_cmap = LinearSegmentedColormap.from_list('pink_blue_cmap', ['#55CDFC', '#FFFFFF', '#F7A8B8'])

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
from reproject import reproject_interp

#from numpy.linalg import lstsq

from scipy.stats import binned_statistic_2d, kstest, ks_2samp, spearmanr
from scipy.optimize import curve_fit

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from calculation_scripts import sfr_ms, distance_from_ms, calc_color
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90, bgs_sii_ne_snr_cut)
from sample_masks import bgs_ne_snr_cut, bgs_oii_ne_snr_cut, bgs_oii_ne_snr_cut, get_galaxy_type_mask

def lightfraction():
    flux_g = CC.catalog['FLUX_G'][BGS_MASK]
    flux_r = CC.catalog['FLUX_R'][BGS_MASK]
    apflux_g = CC.catalog['FIBERFLUX_G'][BGS_MASK]
    apflux_r = CC.catalog['FIBERFLUX_R'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]

    sample_mask = BGS_SNR_MASK
    lo_sample_mask_redshift = BGS_SNR_MASK & (sfr > SFR50) & (mass > M50)
    hi_sample_mask_redshift = BGS_SNR_MASK & (sfr > SFR90) & (mass > M90)
    lo_sample_mask_mass = BGS_SNR_MASK & (sfr > SFR50) & (z < Z50)
    hi_sample_mask_mass = BGS_SNR_MASK & (sfr > SFR90) & (z < Z90)

    lf_g = apflux_g / flux_g
    lf_r = apflux_r / flux_r

    lf_avg = (lf_g + lf_r) * 0.5

    # lightfrac vs redshift
    # --- Figure and grid layout ---
    fs = 16
    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 8),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'right': 0.86}  # leave space on right for colorbar
    )

    h1 = axes[0].hist2d(
        z[lo_sample_mask_redshift],
        lf_avg[lo_sample_mask_redshift],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )
    axes[0].set_xlim(0, 0.4)
    #axes[0].set_ylim(1, 3)
    axes[0].set_ylabel(r'$F_{fiber} / F_{total}$', fontsize=fs)
    #axes[0].tick_params(labelbottom=True)  # show x ticks but not label
    axes[0].set_xlabel("")  # no x label
    axes[0].vlines(Z50, -10, 10, color='k', label=r'redshift upper limit')
    #axes[0].legend()
    axes[0].text(
        0.01, 0.98,
        "low-z sample",
        fontsize=fs, ha='left', va='top',
        transform=axes[0].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2)
    )
    """
    # Add internal title text
    axes[0].text(
        8.02, 2.93,
        f"$\log{{M_\\star}} < {mcenter:.1f}$",
        fontsize=fs-2, ha='left', va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    axes[0].text(
        8.02, 2.76,
        f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}',
        fontsize=fs - 6, va='top', ha='left',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    """

    h2 = axes[1].hist2d(
        z[hi_sample_mask_redshift],
        lf_avg[hi_sample_mask_redshift],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )

    axes[1].set_xlim(0, 0.4)
    #axes[1].set_ylim(1, 3)
    axes[1].set_xlabel(r'$Z$', fontsize=fs)
    axes[1].set_ylabel(r'$F_{fiber} / F_{total}$', fontsize=fs)
    axes[1].vlines(Z90, -10, 10, color='k', label=r'redshift upper limit')
    axes[1].legend(loc="lower left")

    axes[1].text(
        0.01, 0.98,
        "all-z sample",
        fontsize=fs, ha='left', va='top',
        transform=axes[1].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2)
    )
    """
    axes[1].text(
        8.02, 2.93,
        f"$\log{{M_\\star}} \geq {mcenter:.1f}$",
        fontsize=fs-2, ha='left', va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    axes[1].text(
        8.02, 2.76,
        f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}',
        fontsize=fs - 6, va='top', ha='left',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    """

    # --- Keep identical tick positions but hide only the top label on the bottom panel ---
    # Get the tick positions (shared because sharey=True)
    yticks = axes[0].get_yticks()
    # Ensure both axes use the same tick positions
    axes[0].set_yticks(yticks)
    axes[1].set_yticks(yticks)

    # Hide only the top-most y tick LABEL on the bottom axes:
    bottom_yticklabels = axes[1].get_yticklabels()
    if bottom_yticklabels:
        bottom_yticklabels[-1].set_visible(False)

    # Place colorbar axes a bit to the right of the subplots area
    cbar_ax = fig.add_axes([0.89, 0.12, 0.03, 0.755])  # [left, bottom, width, height] in figure coords
    fig.colorbar(h1[3], cax=cbar_ax, label="count")

    plt.show()




    # lightfrac vs mass
    # --- Figure and grid layout ---
    fs = 16
    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 8),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'right': 0.86}  # leave space on right for colorbar
    )

    h1 = axes[0].hist2d(
        mass[lo_sample_mask_mass],
        lf_avg[lo_sample_mask_mass],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )
    axes[0].set_xlim(8, 11.5)
    #axes[0].set_ylim(1, 3)
    axes[0].set_ylabel(r'$F_{fiber} / F_{total}$', fontsize=fs)
    #axes[0].tick_params(labelbottom=True)  # show x ticks but not label
    axes[0].set_xlabel("")  # no x label
    axes[0].vlines(M50, -10, 10, color='k', label=r'$M_\star$ lower limit')
    #axes[0].legend()
    axes[0].text(
        0.01, 0.98,
        "low-z sample",
        fontsize=fs, ha='left', va='top',
        transform=axes[0].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2)
    )

    h2 = axes[1].hist2d(
        mass[hi_sample_mask_mass],
        lf_avg[hi_sample_mask_mass],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )

    axes[1].set_xlim(8, 11.5)
    #axes[1].set_ylim(1, 3)
    axes[1].set_xlabel(r'$\log{M_\star / M_\odot}$', fontsize=fs)
    axes[1].set_ylabel(r'$F_{fiber} / F_{total}$', fontsize=fs)
    axes[1].vlines(M90, -10, 10, color='k', label=r'$M_\star$ lower limit')
    axes[1].legend(loc="lower left")

    axes[1].text(
        0.01, 0.98,
        "all-z sample",
        fontsize=fs, ha='left', va='top',
        transform=axes[1].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2)
    )

    # --- Keep identical tick positions but hide only the top label on the bottom panel ---
    # Get the tick positions (shared because sharey=True)
    yticks = axes[0].get_yticks()
    # Ensure both axes use the same tick positions
    axes[0].set_yticks(yticks)
    axes[1].set_yticks(yticks)

    # Hide only the top-most y tick LABEL on the bottom axes:
    bottom_yticklabels = axes[1].get_yticklabels()
    if bottom_yticklabels:
        bottom_yticklabels[-1].set_visible(False)

    # Place colorbar axes a bit to the right of the subplots area
    cbar_ax = fig.add_axes([0.89, 0.12, 0.03, 0.755])  # [left, bottom, width, height] in figure coords
    fig.colorbar(h1[3], cax=cbar_ax, label="count")

    plt.show()

    print(sum(np.array(lo_sample_mask_mass)), sum(np.array(hi_sample_mask_mass)))


def sdss_probe():
    cat_dir = os.path.expanduser('~') + '/Downloads/'

    print("reading in table")
    lines = Table.read(cat_dir + 'gal_line_dr7_v5_2.fit.gz')
    info  = Table.read(cat_dir + 'gal_info_dr7_v5_2.fit.gz')

    """
    # 1) outer join on both columns
    merged = join(lines, info, keys=['PLATEID', 'FIBERID'], join_type='outer')

    # 2) replace masked (missing) values with sensible blanks:
    for name in merged.colnames:
        col = merged[name]
        # only do something if column is masked or contains masked values
        if getattr(col, 'mask', False) is not False and getattr(col, 'mask', None) is not None:
            if col.dtype.kind in ('U', 'S', 'O'):  # string/object-like
                merged[name] = col.filled('')  # empty string for missing strings
            else:
                merged[name] = col.filled(np.nan)  # NaN for numeric types

    # merged now contains all unique (PLATEID,FIBERID) pairs with blanks where a side was missing

    sdssCat = merged
    """
    common_cols = []
    for name in lines.colnames:
        if name in info.colnames and np.all(lines[name] == info[name]):
            common_cols.append(name)

    info_unique = info[[c for c in info.colnames if c not in common_cols]]
    sdssCat = hstack([lines, info_unique])

    #print("writing out table")
    #sdssCat.write(f'{cat_dir}merged_sdss_table.fits', overwrite=True)

    # keep only unique rows based on the combination of 'RA' and 'DEC'
    sdssCat = unique(sdssCat, keys=['RA', 'DEC'])

    #print(sdssCat.colnames)
    #print(sdssCat['FIBERID'][:20])

    snr_lim = 5

    sii_16_snr = np.array(sdssCat['SII_6717_FLUX']) / np.array(sdssCat['SII_6717_FLUX_ERR'] * 1.621)
    sii_31_snr = np.array(sdssCat['SII_6731_FLUX']) / np.array(sdssCat['SII_6731_FLUX_ERR'] * 1.621)
    ha_snr = np.array(sdssCat['H_ALPHA_FLUX']) / np.array(sdssCat['H_ALPHA_FLUX_ERR'] * 2.473)
    hb_snr = np.array(sdssCat['H_BETA_FLUX']) / np.array(sdssCat['H_BETA_FLUX_ERR'] * 1.882)
    z = np.array(sdssCat['Z'])
    z_err = np.array(sdssCat['Z_ERR']) / z
    z_warn = np.array(sdssCat['Z_WARNING'])
    galaxy_targ = np.array(['GALAXY' in str(x).strip()  # convert bytes to str and strip whitespace
        for x in sdssCat['TARGETTYPE']])

    source_mask = ((sii_16_snr > snr_lim) &
                   (sii_31_snr > snr_lim) &
                   (ha_snr > snr_lim) &
                   (hb_snr > snr_lim) &
                   (z < 0.2) &
                   (z_err < 0.05) &
                   (z_warn == 0) &
                   (galaxy_targ)
                   )

    print(sum(np.array(source_mask)))

    print(sii_16_snr[source_mask][:10])


def check_sample_properties():
    oiii_snr = CC.catalog['OIII_4363_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OIII_4363_FLUX_IVAR'][BGS_MASK])

    print(sum(np.array(oiii_snr[LO_Z_MASK] > 3)))
    print(sum(np.array(LO_Z_MASK)))
    print(sum(np.array(oiii_snr[HI_Z_MASK] > 3)))
    print(sum(np.array(HI_Z_MASK)))


def oii_sii_vs_metallicity(sample_mask=BGS_SNR_MASK):

    metallicity = np.array(CC.catalog['METALLICITY_O3N2'][BGS_MASK])
    ne_oii = np.array(CC.catalog['NE_OII'][BGS_MASK])
    ne_sii = np.array(CC.catalog['NE_SII'][BGS_MASK])

    ne_ratio = ne_oii / ne_sii

    oiii_5007_flux = np.array(CC.catalog['OIII_5007_FLUX'][BGS_MASK])
    oiii_5007_err_inv = np.array(np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK]))
    nii_6584_flux = np.array(CC.catalog['NII_6584_FLUX'][BGS_MASK])
    nii_6584_err_inv = np.array(np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK]))
    halpha_flux = np.array(CC.catalog['HALPHA_FLUX'][BGS_MASK])
    halpha_flux_err_inv = np.array(np.sqrt(CC.catalog['HALPHA_FLUX_IVAR'][BGS_MASK]))
    hbeta_flux = np.array(CC.catalog['HBETA_FLUX'][BGS_MASK])
    hbeta_flux_err_inv = np.array(np.sqrt(CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK]))

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    oiii_5007_snr = oiii_5007_flux * oiii_5007_err_inv
    nii_6584_snr = nii_6584_flux * nii_6584_err_inv
    halpha_snr = halpha_flux * halpha_flux_err_inv
    hbeta_snr = hbeta_flux * hbeta_flux_err_inv

    snr_lim = 3

    # This is just the metallicity lines
    metallicity_mask = generate_combined_mask(oiii_5007_snr > snr_lim, nii_6584_snr > snr_lim, halpha_snr > snr_lim, hbeta_snr > snr_lim)
    hii_galaxy_mask, agn_galaxy_mask, _, _ = get_galaxy_type_mask()

    sample_mask = (sample_mask) & (metallicity_mask) & (~agn_galaxy_mask)

    fs = 18

    plt.hist2d(metallicity[sample_mask], np.log10(ne_ratio[sample_mask]), bins=(45, 70), norm=mpl.colors.LogNorm())
    plt.xlabel(r'$12 + \log{O/H}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e([OII]) / n_e([SII])}$', fontsize=fs)
    plt.xlim(8, 9)
    plt.ylim(-1.5, 1.5)
    plt.colorbar(label='counts')
    plt.savefig(f'plots_for_proposal/ne_ratio_vs_metallicity.png', dpi=150)
    plt.show()


def checking_values():

    a_ha = CC.catalog['A_HALPHA'][BGS_MASK]
    a_ha_lo = a_ha[LO_Z_MASK]
    a_ha_hi = a_ha[HI_Z_MASK]

    print(np.median(a_ha_lo), np.median(a_ha_hi))



def sii_ratio_maps(galname):

    # ===========================
    # Step 1: Read in the FITS files
    # ===========================
    f1 = fits.open(f'spectrum_maps/{galname}/{galname}_f_SII6717.fits')
    f2 = fits.open(f'spectrum_maps/{galname}/{galname}_f_SII6731.fits')

    data1 = f1[0].data
    data2 = f2[0].data

    wcs1 = WCS(f1[0].header)
    wcs2 = WCS(f2[0].header)

    # ===========================
    # Step 2: Reproject data2 onto data1’s WCS grid
    # ===========================
    # This ensures spatial alignment in FK5 coordinates
    data2_reproj, footprint = reproject_interp((data2, wcs2), wcs1, shape_out=data1.shape)

    # ===========================
    # Step 3: Bin (average) the data to improve SNR
    # ===========================
    bin_size = 3  # <-- adjust to 2, 3, 4, etc. for more smoothing
    data1_binned = block_reduce(data1, block_size=bin_size, func=np.nanmean)
    data2_binned = block_reduce(data2_reproj, block_size=bin_size, func=np.nanmean)

    # Adjust WCS to match the binned image scale
    wcs_binned = wcs1.deepcopy()
    if wcs_binned.wcs.has_cd():
        wcs_binned.wcs.cd *= bin_size
    else:
        wcs_binned.wcs.cdelt *= bin_size

    # ===========================
    # Step 4: Compute the ratio map
    # ===========================
    ratio = np.full_like(data1_binned, np.nan, dtype=float)
    valid = (data1_binned > 0) & (data2_binned > 0)
    ratio[valid] = data1_binned[valid] / data2_binned[valid]

    # ===========================
    # Step 5: Plot the ratio map
    # ===========================
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(projection=wcs_binned)
    im = ax.imshow(ratio, origin='lower', cmap='magma',
                   vmin=np.nanpercentile(ratio, 5),
                   vmax=np.nanpercentile(ratio, 95))
    cb = plt.colorbar(im, ax=ax, label=r'$\lambda 6717 / \lambda 6731$')

    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('Dec (J2000)')
    ax.set_title(f'{galname} [S II] Ratio Map (binned {bin_size}×{bin_size})')

    plt.tight_layout()
    plt.savefig(f'plots_for_proposal/ratio_maps_{galname}.png', dpi=250)
    plt.show()


if __name__ == '__main__':
    #oii_sii_vs_metallicity(sample_mask=HI_Z_MASK)
    #checking_values()
    galaxies = ["eso498-g5","eso499-g37","ic2560","ic5273","ngc0289","ngc0337","ngc1042","ngc1084","ngc1097","ngc1309",
                "ngc1483","ngc1512","ngc2104","ngc2835","ngc3081","ngc3256","ngc3393","ngc3513","ngc3521","ngc3783",
                "ngc4030","ngc4517a","ngc4592","ngc4593","ngc4603","ngc4790","ngc4900","ngc4941","ngc4980","ngc5334",
                "ngc5584","ngc5643","ngc5806","ngc7162","ngc7421","ngc7496","ngc7552","pgc3853"]
    for gal in galaxies:
        sii_ratio_maps(gal)