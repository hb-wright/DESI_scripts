import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from astropy.io import ascii

plt.rcParams.update({
    'text.usetex': True
})
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
# Create custom colormaps for plots
pink_blue_2val_cmap = LinearSegmentedColormap.from_list('pink_blue_cmap', ['#55CDFC', '#FFFFFF', '#F7A8B8'])

import numpy as np

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


def histogram_plots():
    """
    This makes the 6x2 histogram plot for the paper
    :return:
    """
    redshift = CC.catalog['Z'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mstar = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)
    sfr_sd_bgs = CC.catalog['SFR_SD'][BGS_MASK]
    metallicity = CC.catalog['METALLICITY_R23'][BGS_MASK]

    # Create a figure with 6 rows and 2 columns
    fig, axes = plt.subplots(6, 2, figsize=(10, 12))

    # Adjust the space between subplots for better readability
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Store the data in a 2D list (or array)
    data = [
        [redshift[LO_Z_MASK], redshift[HI_Z_MASK]],
        [mstar[LO_Z_MASK], mstar[HI_Z_MASK]],
        [sfr[LO_Z_MASK], sfr[HI_Z_MASK]],
        [sfr_sd_bgs[LO_Z_MASK], sfr_sd_bgs[HI_Z_MASK]],
        [ne[LO_Z_MASK], ne[HI_Z_MASK]],
        [metallicity[LO_Z_MASK], metallicity[HI_Z_MASK]]
    ]

    xlabels = [
        r'$z$',
        r'$\log\,(\,M_\star\,[M_\odot]\,)$',
        r'$\log\,(\mathrm{SFR}\,[M_\odot\,\mathrm{yr}^{-1}])$',
        r'$\log\,(\Sigma_{\mathrm{SFR}}\,[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}])$',
        r'$\log\,(n_e\,[\mathrm{cm}^{-3}])$',
        r'$12 + \log\,(\mathrm{O}/\mathrm{H})$'
    ]

    xlimits = [
        (0, 0.35),
        (9, 11.5),
        (-0.1, 2),
        (-2.2, 0.1),
        (1, 3),
        (8, 9.3)
    ]

    # Loop over each subplot
    for i in range(6):
        for j in range(2):
            ax = axes[i, j]

            current_data = data[i][j]

            # Create the histogram
            ax.hist(current_data, bins=50, color='black', histtype='step', linestyle='-', linewidth=1.5)

            # Set axis labels and title (adjust as needed)
            if j == 0:  # Left column for y-axis labels
                ax.set_ylabel(r'$N$', fontsize=12)
            ax.set_xlabel(xlabels[i], fontsize=12)
            ax.set_xlim(xlimits[i])

            # Set titles for each subplot (adjust as needed)
            # ax.set_title(f'Histogram {i * 2 + j + 1}', fontsize=13)

            # Set grid and ticks
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, colors='black')

            # Set tick label font size
            ax.tick_params(axis='both', labelsize=10)

            # Remove top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    axes[0, 0].set_title('low-$z$', fontsize=14)
    axes[0, 1].set_title('all-$z$', fontsize=14)
    plt.savefig(f'paper_figures/histograms.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def plot_ne_distribution(sample_mask=BGS_SNR_MASK):
    """
    This plots a histogram of the ne distribution for the given sample mask
    :param snr_mask: default mask is the full snr>5 sample at all redshifts
    :return: none
    """

    ne, _ = bgs_ne_snr_cut()
    ne_oii, _ = bgs_oii_ne_snr_cut()
    ne_sii, _ = bgs_sii_ne_snr_cut()

    sample = 0
    sample_str = "custom sample"
    if sample_mask is BGS_SNR_MASK:
        sample = 1
        sample_str = "all galaxies"
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_str = "low-z"
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_str = "all-z"

    fs = 20

    # Statistics:
    #print("Median:", np.median(ne[sample_mask]))
    #print("Mean:", np.average(ne[sample_mask]))
    #print("Stdev:", np.std(ne[sample_mask]))

    plt.hist(ne_oii[sample_mask], bins=50)
    plt.xlim(0, 3.5)
    if sample == 2:
        tit = f"Electron density distribution (low-z, {len(ne[sample_mask])} galaxies)"
    elif sample == 3:
        tit = f"Electron density distribution (all-z, {len(ne[sample_mask])} galaxies)"
    else:
        tit = f"Electron density distribution ({len(ne[sample_mask])} galaxies)"
    plt.title(tit, fontsize=fs-4)
    plt.xlabel(r"$\log{n_e([OII])/cm^{3}}$", fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-4)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"paper_figures/ne_oii_dist_{sample}.png", dpi=PLOT_DPI)
    plt.show()

    plt.hist(ne_sii[sample_mask], bins=50)
    plt.xlim(0, 3.5)
    if sample == 2:
        tit = f"Electron density distribution (low-z, {len(ne[sample_mask])} galaxies)"
    elif sample == 3:
        tit = f"Electron density distribution (all-z, {len(ne[sample_mask])} galaxies)"
    else:
        tit = f"Electron density distribution ({len(ne[sample_mask])} galaxies)"
    plt.title(tit)
    plt.xlabel(r"$\log{n_e([SII])/cm^{3}}$", fontsize=fs)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    # ne(OII) vs ne(SII)
    spearcorr = spearmanr(ne_oii[sample_mask], ne_sii[sample_mask])
    plt.hist2d(ne_oii[sample_mask], ne_sii[sample_mask], bins=80, norm=mpl.colors.LogNorm())
    plt.title(f"$n_e$ from different ions, {sample_str}")
    plt.xlim(0.5, 3.4)
    plt.ylim(0.5, 3.4)
    plt.text(0.015, 0.985, f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}', transform=plt.gca().transAxes, fontsize=fs-6, va='top', ha='left')
    plt.xlabel(r"$\log{n_e([OII])/cm^{-3}}$", fontsize=fs)
    plt.ylabel(r"$\log{n_e([SII])/cm^{-3}}$", fontsize=fs)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    ne_oii_sii_ks = ks_2samp(np.array(ne_oii[sample_mask]), np.array(ne_sii[sample_mask]))

    fig, ax = plt.subplots()

    counts, edges = np.histogram(ne_oii[sample_mask], bins=100)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label= "$n_e([OII])$")
    counts, edges = np.histogram(ne_sii[sample_mask], bins=100)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label= "$n_e([SII])$")
    plt.xlabel(r"$\log{n_e/cm^{-3}}$", fontsize=fs)

    ks_string = f"K-S test: p = {ne_oii_sii_ks.pvalue:.3e}"
    plt.text(0.02, 0.98, ks_string, ha='left', va='top', transform=ax.transAxes, fontsize=fs - 2)
    #plt.legend(loc='lower right')
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/ne_ks.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()



def compare_sfr(sample_mask=BGS_SNR_MASK):

    sfr_cigale = np.array(CC.catalog['SFR_CIGALE'][BGS_MASK])
    sfr_ha = np.array(CC.catalog['SFR_HALPHA'][BGS_MASK])
    mstar = np.array(CC.catalog['MSTAR_CIGALE'][BGS_MASK])
    z = np.array(CC.catalog['Z'][BGS_MASK])

    sample = 0
    clr = 'k'
    mlim = 4
    redshift_sample_mask = BGS_SNR_MASK
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        sfrlim = SFR50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        sfrlim = SFR90
        clr = 'r'

    sample_mask = generate_combined_mask(redshift_sample_mask, mstar >= mlim)

    sfr_cigale = sfr_cigale[sample_mask]
    sfr_ha = sfr_ha[sample_mask]
    """
    # Histograms of each sfr
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(sfr_ha, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(sfr_ha)}\nmedian: {np.median(sfr_ha)}\nstdev: {np.std(sfr_ha)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from $H\alpha$ ($\log{m_\star/m_\odot}$)", xlim=(-8,2.5))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(sfr_cigale, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(sfr_cigale)}\nmedian: {np.median(sfr_cigale)}\nstdev: {np.std(sfr_cigale)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from CIGALE ($\log{m_\star/m_\odot}$)", xlim=(-8,2.5))
    plt.show()

    #print(cigale_sfr)
    #print(f"SFR Avg: {np.average(cigale_sfr)}, stdev: {np.std(cigale_sfr)}")
    #print(halpha_sfr)
    #print(f"SFR Avg: {np.average(halpha_sfr)}, stdev: {np.std(halpha_sfr)}")
    fig = plt.figure(figsize=(6, 5))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.hist2d(sfr_ha, sfr_ha - sfr_cigale, bins=50, norm=mpl.colors.LogNorm())
    ax_main.plot(np.linspace(-10,10, 100), np.zeros(100), color='k')
    ax_main.vlines(sfrlim, -10, 10, color=clr)
    ax_main.set(xlabel=r"$log(SFR(H\alpha)/M_\odot/yr)$", ylabel=r"$\log((SFR(H\alpha) - SFR(CIGALE))/M_\odot/yr)$", xlim=(-1.2,1.8), ylim=(-1,1))

    ax_yDist.hist(sfr_ha - sfr_cigale, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(sfr_ha, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.show()
    """

    fig = plt.figure(figsize=(7, 7))

    # Outer GridSpec: 2 rows (main block + colorbar)
    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[20, 0.5],  # main block tall, colorbar short
        hspace=0.25  # space between them
    )

    # Inner GridSpec: main plot + histograms, tightly packed
    gs_inner = GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs_outer[0],
        hspace=0.0, wspace=0.0  # no gap between top/main/right
    )

    # Axes inside the inner grid
    ax_xDist = fig.add_subplot(gs_inner[0, :3])  # top histogram
    ax_main = fig.add_subplot(gs_inner[1:4, :3])  # main 2D histogram
    ax_yDist = fig.add_subplot(gs_inner[1:4, 3], sharey=ax_main)  # right histogram

    # Colorbar axis in the outer grid, below everything
    ax_cbar = fig.add_subplot(gs_outer[1, 0])

    # 2D histogram
    h = ax_main.hist2d(sfr_ha, sfr_ha - sfr_cigale, bins=(50,80), norm=mpl.colors.LogNorm())

    # Marginal histograms
    ax_yDist.hist(sfr_ha - sfr_cigale, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(sfr_ha, bins=200, orientation='vertical', align='mid')

    # Reference lines
    ax_main.plot(np.linspace(-10, 10, 100), np.zeros(100), color='k')
    ax_main.vlines(sfrlim, -10, 10, color=clr)
    ax_main.set(
        xlabel=r"$\log(SFR(H\alpha)/M_\odot/yr)$",
        ylabel=r"$\log((SFR(H\alpha) - SFR(CIGALE))/M_\odot/yr)$",
        xlim=(-1.2, 1.8),
        ylim=(-1, 1)
    )

    # Right histogram settings
    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()
    ax_yDist.yaxis.set_label_position("right")

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    # Colorbar below everything
    cbar = fig.colorbar(h[3], cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Counts')

    plt.show()


def plot_redshift_vs_mass_sfr():
    """
    Plots mass and sfr vs redshift with completeness limits for each sample.
    Not built to work with custom masks
    :return: none
    """

    mass_bgs = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_bgs = CC.catalog['SFR_HALPHA'][BGS_MASK]
    redshift_bgs = CC.catalog['Z'][BGS_MASK]

    mass = mass_bgs[BGS_SNR_MASK]
    sfr = sfr_bgs[BGS_SNR_MASK]
    redshift = redshift_bgs[BGS_SNR_MASK]

    fs = 16

    print("lo-z median:", np.median(redshift_bgs[LO_Z_MASK]))
    print("hi-z mean:", np.average(redshift_bgs[LO_Z_MASK]))
    print("hi-z median:", np.median(redshift_bgs[HI_Z_MASK]))
    print("hi-z mean:", np.average(redshift_bgs[HI_Z_MASK]))

    """
    fig, ax = plt.subplots()
    plt.hist2d(redshift, mass, bins=60, norm=mpl.colors.LogNorm())
    plt.vlines(Z50, M50, 13, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    plt.hlines(M50, 0, Z50, color='b')
    plt.vlines(Z90, M90, 13, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    plt.hlines(M90, 0, Z90, color='r')
    #plt.title(f'Stellar mass vs redshift ({sum(BGS_SNR_MASK)} galaxies)')
    plt.xlabel("z", fontsize=fs)
    plt.ylabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
    plt.xlim(0, 0.4)
    plt.ylim(7, 11.5)
    plt.colorbar(label="count")
    plt.legend(loc='lower right')
    #ax.text(0.01, 0.98, f'{sum(BGS_SNR_MASK)} galaxies',
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=ax.transAxes, fontsize=fs-4)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_mass_redshift.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()
    
        fig, ax = plt.subplots()
    plt.hist2d(redshift, sfr, bins=60, norm=mpl.colors.LogNorm())
    plt.vlines(Z50, SFR50, 10, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    plt.hlines(SFR50, 0, Z50, color='b', label='_nolegend_')
    plt.vlines(Z90, SFR90, 10, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    plt.hlines(SFR90, 0, Z90, color='r', label='_nolegend_')
    #plt.title(f'SFR vs redshift ({sum(BGS_SNR_MASK)} galaxies)')
    plt.xlabel("z", fontsize=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
    plt.xlim(0, 0.4)
    plt.ylim(-2, 2)
    plt.colorbar(label="count")
    plt.legend(loc='lower right')
    #ax.text(0.01, 0.98, f'{sum(BGS_SNR_MASK)} galaxies',
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=ax.transAxes, fontsize=fs-4)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_sfr_redshift.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()
    
    """

    fs = 18

    # 2D histogram with distributions for mass and redshift

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]

    # Main part of figure
    sp = ax_main.hist2d(redshift, mass, bins=60, cmap="Greys", norm=mpl.colors.LogNorm())
    ax_main.set(xlim=(0, 0.4), ylim=(7, 11.5))
    ax_main.set_xlabel(r"z", fontsize=fs)
    ax_main.set_ylabel(r"$\log{M_\star/M_\odot}$", fontsize=fs)
    ax_main.vlines(Z50, M50, 13, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    ax_main.hlines(M50, 0, Z50, color='b')
    ax_main.vlines(Z90, M90, 13, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    ax_main.hlines(M90, 0, Z90, color='r')
    ax_main.legend(loc=4)

    bins = np.histogram_bin_edges(np.concatenate([mass, mass_bgs[LO_Z_MASK], mass_bgs[HI_Z_MASK]]), bins=80)

    ax_yDist.hist(mass, bins=bins, orientation='horizontal', align='mid', color='k', alpha=0.3)
    ax_yDist.hist(mass_bgs[LO_Z_MASK], bins=bins, orientation='horizontal', align='mid', color='b', alpha=0.3)
    ax_yDist.hist(mass_bgs[HI_Z_MASK], bins=bins, orientation='horizontal', align='mid', color='r', alpha=0.3)
    ax_yDist.set_xticks([400, 800])

    bins = np.histogram_bin_edges(np.concatenate([redshift, redshift_bgs[LO_Z_MASK], redshift_bgs[HI_Z_MASK]]), bins=80)

    ax_xDist.hist(redshift, bins=bins, orientation='vertical', align='mid', color='k', alpha=0.3)
    ax_xDist.hist(redshift_bgs[LO_Z_MASK], bins=bins, orientation='vertical', align='mid', color='b', alpha=0.3)
    ax_xDist.hist(redshift_bgs[HI_Z_MASK], bins=bins, orientation='vertical', align='mid', color='r', alpha=0.3)
    yticks = ax_xDist.get_yticks()
    yticklabels = ["" if t == 0 else f"{int(t)}" for t in yticks]
    ax_xDist.set_yticklabels(yticklabels)

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_mass_redshift.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # 2D histogram with distributions for sr and redshift

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)  # , top=0.95)
    axs = [ax_main, ax_yDist]  # , ax_xDist]

    # Main part of figure
    sp = ax_main.hist2d(redshift, sfr, bins=60, cmap="Greys", norm=mpl.colors.LogNorm())
    ax_main.set(xlim=(0, 0.4), ylim=(-2, 2))
    ax_main.set_xlabel(r"z", fontsize=fs)
    ax_main.set_ylabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
    ax_main.vlines(Z50, SFR50, 10, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    ax_main.hlines(SFR50, 0, Z50, color='b')
    ax_main.vlines(Z90, SFR90, 10, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    ax_main.hlines(SFR90, 0, Z90, color='r')
    ax_main.legend(loc=4)

    bins = np.histogram_bin_edges(np.concatenate([sfr, sfr_bgs[LO_Z_MASK], sfr_bgs[HI_Z_MASK]]), bins=80)

    ax_yDist.hist(sfr, bins=bins, orientation='horizontal', align='mid', color='k', alpha=0.3)
    ax_yDist.hist(sfr_bgs[LO_Z_MASK], bins=bins, orientation='horizontal', align='mid', color='b', alpha=0.3)
    ax_yDist.hist(sfr_bgs[HI_Z_MASK], bins=bins, orientation='horizontal', align='mid', color='r', alpha=0.3)
    ax_yDist.set_xticks([400, 800])

    bins = np.histogram_bin_edges(np.concatenate([redshift, redshift_bgs[LO_Z_MASK], redshift_bgs[HI_Z_MASK]]), bins=80)

    ax_xDist.hist(redshift, bins=bins, orientation='vertical', align='mid', color='k', alpha=0.3)
    ax_xDist.hist(redshift_bgs[LO_Z_MASK], bins=bins, orientation='vertical', align='mid', color='b', alpha=0.3)
    ax_xDist.hist(redshift_bgs[HI_Z_MASK], bins=bins, orientation='vertical', align='mid', color='r', alpha=0.3)
    yticks = ax_xDist.get_yticks()
    yticklabels = ["" if t == 0 else f"{int(t)}" for t in yticks]
    ax_xDist.set_yticklabels(yticklabels)

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_sfr_redshift.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def plot_sfr_ms(sample_mask=BGS_SNR_MASK, plot=True):

    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mstar = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    clr = 'k'
    mlim = 0
    sfrlim = -100
    zlim = 1
    redshift_sample_mask = BGS_SNR_MASK
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        sfrlim = SFR50
        zlim = 0.1
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        sfrlim = SFR90
        zlim = 0.2
        clr = 'r'

    ms_sample_mask = generate_combined_mask(redshift_sample_mask, mstar >= mlim)#, z <= zlim)
    #print(sum(np.array(ms_sample_mask)))
    fs = 18

    #o1, o2, c = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], 2)
    #o1, c = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], 1)

    coeffs = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], deg=2)

    # Evaluate the polynomial
    p = np.poly1d(coeffs)

    if plot:
        # x-axis arrays for our fit (complete [1] and incomplete[2] regions)
        x1 = np.linspace(mlim,20,100)
        x2 = np.linspace(0, mlim, 100)
        # full x-axis array
        xt = np.linspace(0, 20, 200)
        # x-axis for Whitaker+14, broken power law
        cmass_whitaker = 10.2
        xt_whit_lo = np.linspace(0, cmass_whitaker, 100)
        xt_whit_hi = np.linspace(cmass_whitaker, 20, 100)

        mstar_wht = np.array([9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1])
        loga = np.array([-9.54, -9.50, -9.54, -9.58, -9.69, -9.93, -10.11, -10.28, -10.53, -10.65])
        b = np.array([1.95, 1.86, 1.90, 1.98, 2.16, 2.63, 2.88, 3.03, 3.37, 3.45])

        yt_whitaker_zcorr_loz = loga * mstar_wht * np.log10(1 + 0.141) ** b
        yt_whitaker_zcorr_hiz = loga * mstar_wht * np.log10(1 + 0.237) ** b
        #print(yt_whitaker_zcorr_hiz)

        # Our fit
        y1 = p(x1)
        y2 = p(x2)

        # Galaxy ages calculated with Ned Wright's Cosmology calculator using flat cosmology and mean redshifts for each sample:
        # hi-z mean: 0.141
        # hi-z mean: 0.237
        age_loz = 11.924
        age_hiz = 10.885
        yt_speagle_loz = (0.84 - 0.026*age_loz) * xt - (6.51 - 0.11*age_loz)
        yt_speagle_hiz = (0.84 - 0.026*age_hiz) * xt - (6.51 - 0.11*age_hiz)

        # Schreiber+15
        r_loz = np.log10(1 + 0.141)
        r_hiz = np.log10(1 + 0.237)
        xt_proc_loz = (xt - 9) - 0.36 - 2.5*r_loz
        xt_proc_hiz = (xt - 9) - 0.36 - 2.5*r_hiz
        yt_schreiber_loz = (xt - 9) - 0.5 + 1.5*r_loz - 0.3*(np.where(xt_proc_loz<0, 0, xt_proc_loz)**2)
        yt_schreiber_hiz = (xt - 9) - 0.5 + 1.5*r_hiz - 0.3*(np.where(xt_proc_hiz<0, 0, xt_proc_hiz)**2)

        # This lets us calculate the offset between Whitaker and Schreiber
        r_whit = np.log10(1 + 0.75)
        xt_proc_loz_sv = (10 - 9) - 0.36 - 2.5 * r_loz
        xt_proc_hiz_sv = (10 - 9) - 0.36 - 2.5 * r_hiz
        xt_proc_whit_sv = (10 - 9) - 0.36 - 2.5 * r_whit
        yt_schreiber_loz_sv = (10 - 9) - 0.5 + 1.5 * r_loz - 0.3 * (max(0, xt_proc_loz_sv) ** 2)
        yt_schreiber_hiz_sv = (10 - 9) - 0.5 + 1.5 * r_hiz - 0.3 * (max(0, xt_proc_hiz_sv) ** 2)
        yt_schreiber_whit_sv = (10 - 9) - 0.5 + 1.5 * r_hiz - 0.3 * (max(0, xt_proc_whit_sv) ** 2)

        loz_offset = yt_schreiber_whit_sv - yt_schreiber_loz_sv
        hiz_offset = yt_schreiber_whit_sv - yt_schreiber_hiz_sv

        # Whitaker has both a 2nd order polynomial fit and a broken power law
        yt_whitaker_p2_loz = -27.40 + 5.02 * xt + -0.22 * xt ** 2 - loz_offset
        yt_whitaker_p2_hiz = -27.40 + 5.02 * xt + -0.22 * xt ** 2 - hiz_offset
        # This is the 2nd order polynomial, we aren't including it
        # yt_whitaker_bpl_lo = 0.94 * (xt_whit_lo - 10.2) + 1.11
        # yt_whitaker_bpl_hi = 0.14 * (xt_whit_hi - 10.2) + 1.11

        fig, ax = plt.subplots(figsize=(6, 5))
        plt.hist2d(mstar[redshift_sample_mask], sfr[redshift_sample_mask], bins=(80,40), norm=mpl.colors.LogNorm())
        #plt.plot(xt, yt_speagle, label='Speagle+14', color='tab:blue')
        #plt.plot(xt_whit_lo, yt_whitaker_bpl_lo, color='tab:purple')
        #plt.plot(xt_whit_hi, yt_whitaker_bpl_hi, color='tab:purple')
        #plt.plot(xt, yt_schreiber, color='tab:green', label='Schreiber+15')
        #plt.plot(x, y2, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.ylim(-1.5, 2)
        plt.colorbar(label='count')
        plt.plot(x1, y1, color='k', label='our polynomial fit', linewidth=3)
        plt.plot(x2, y2, color='k', linestyle='--', linewidth=3, label='_nolegend_')
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        alpha = 0.3
        color = 'gray'
        zorder = 0

        x_cut = mlim
        y_cut = sfrlim

        rect_left = Rectangle(
            (x_cut, ymin),
            x_cut - xmax,
            ymax - ymin,
            facecolor=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(rect_left)

        rect_bottom_right = Rectangle(
            (x_cut, ymin),
            xmax,
            y_cut - ymin,
            facecolor=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(rect_bottom_right)

        plt.vlines(mlim, sfrlim, 13, color=clr)
        plt.hlines(sfrlim, 100, mlim, color=clr)

        if sample == 2:
            ax.text(0.02, 0.98, f'low-z',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes, fontsize=fs - 4,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.5,
                        edgecolor='none',
                        boxstyle="round,pad=0.3,rounding_size=.3")
                    )
            # plt.plot(xt, yt_speagle_loz, label='Speagle+14', color='tab:blue')
            plt.plot(xt, yt_schreiber_loz, color='tab:green', label='Schreiber+15')
            # plt.plot(mstar_wht, yt_whitaker_zcorr_loz, color='tab:purple', label='Whitaker+14')
            plt.plot(xt, yt_whitaker_p2_loz, label=r'Whitaker+14 (shifted)', color='tab:purple')
        if sample == 3:
            ax.text(0.02, 0.98, f'all-z',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes, fontsize=fs-4,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.5,
                        edgecolor='none',
                        boxstyle="round,pad=0.3,rounding_size=.3")
                    )
            #plt.plot(xt, yt_speagle_hiz, label='Speagle+14', color='tab:blue')
            plt.plot(xt, yt_schreiber_hiz, color='tab:green', label='Schreiber+15')
            #plt.plot(mstar_wht, yt_whitaker_zcorr_hiz, color='tab:purple', label='Whitaker+14')
            plt.plot(xt, yt_whitaker_p2_hiz, label=r'Whitaker+14 (shifted)', color='tab:purple')
        plt.xlabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
        plt.ylabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
        plt.legend(loc='lower right')
        if PLOT_SAVE:
            plt.savefig(f'paper_figures/paper_sfr_ms_sample_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
        plt.show()
        """
        plt.hist2d(mstar, specific_sfr, bins=(200,70), norm=mpl.colors.LogNorm())
        #plt.plot(x, np.ones(len(x))*sSFR_cut, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.colorbar(label='count')
        #plt.ylim(-.35, .35)
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r'$\log{SFR / M_\star}$')
        #plt.legend(loc='upper left')
        plt.show()
        """

    return p

def plot_redshift_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Plots electron density vs redshift
    :return: none
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    redshift = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)  # these are both bgs length

    sample = 0
    zlim=0.4
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = np.array((BGS_SNR_MASK) & (sfr >= SFR50) & (mass >= M50))
        zlim = Z50
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = np.array((BGS_SNR_MASK) & (sfr >= SFR90) & (mass >= M90))
        zlim = Z90

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    ne = ne[sample_mask]
    redshift = redshift[sample_mask]

    fs = 16

    p, V = np.polyfit(redshift[redshift <= zlim], ne[redshift <= zlim], 1, cov=True)
    m = p[0]
    dm = np.sqrt(V[0][0])
    b = p[1]
    fit_x = np.linspace(0, zlim,10)
    rest_x = np.linspace(zlim,.5,10)
    fit_y = m * fit_x + b
    rest_y = m * rest_x + b

    fig, ax = plt.subplots()
    plt.hist2d(redshift, ne, bins=60, cmap='viridis', norm=mpl.colors.LogNorm())
    plt.plot(fit_x, fit_y, color='k', label='linear fit (slope = {:.3f}'.format(m) + ' $\pm$ {:.3f})'.format(dm))
    plt.plot(rest_x, rest_y, color='k', linestyle='--')
    if sample == 2:
        plt.vlines(Z50, 0, 3.5, color='b')#, label="Completeness upper limit")
        #plt.title(f'Electron density vs redshift (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.02, 0.98, f'low-z',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4,
                bbox=dict(
                    facecolor='white',
                    alpha=0.5,
                    edgecolor='none',
                    boxstyle="round,pad=0.3,rounding_size=.3")
                )
    elif sample == 3:
        plt.vlines(Z90, 0, 3.5, color='r')#, label="Completeness upper limit")
        #plt.title(f'Electron density vs redshift (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.02, 0.98, f'all-z',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4,
                bbox=dict(
                    facecolor='white',
                    alpha=0.5,
                    edgecolor='none',
                    boxstyle="round,pad=0.3,rounding_size=.3")
                )
    else:
        #plt.title(f'Electron density vs redshift ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlabel("$z$", fontsize=fs)
    plt.ylabel(r'$\log\,(n_e\,[\mathrm{cm}^{-3}])$', fontsize=fs)
    plt.legend(loc='lower left')
    plt.xlim(0, 0.4)
    plt.ylim(0, 3.5)
    # After you have defined zlim and set x-limits
    xmax = ax.get_xlim()[1]
    ax.axvspan(
        zlim, xmax,
        facecolor='gray',
        alpha=0.3,
        zorder=0  # behind points/lines
    )
    plt.colorbar(label="count")
    plt.tight_layout()
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_ne_redshift_sample_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def plot_sfr_vs_mass_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Generates a plot of sfr vs mass with each bin color-coded by median ne
    :param sample_mask: Optional extra mask that is placed after snr cuts
    :return:
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        clr = 'r'

    mass_sample = mass[sample_mask]
    sfr_sample = sfr[sample_mask]
    ne_sample = ne[sample_mask]

    fs = 20

    o1, o2, c = plot_sfr_ms(sample_mask=sample_mask, plot=False)

    #x_ms = np.linspace(0,20,100)
    #y_ms = o1 * x_ms**2 + o2 * x_ms + c

    x1 = np.linspace(mlim, 20, 100)
    x2 = np.linspace(0, mlim, 100)
    y1 = o1 * x1 ** 2 + o2 * x1 + c
    y2 = o1 * x2 ** 2 + o2 * x2 + c

    # Define the number of bins
    x_bins = 80
    y_bins = 40

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic='median', bins=[x_bins, y_bins]
    )

    figdim = (8, 5)
    # Plot the result
    fig, ax = plt.subplots(figsize=figdim)
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    #plt.plot(x1, y1, color=clr, label='polynomial fit')
    #plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z',#, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z',#, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" ({sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfr_vs_mstar_vs_ne_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=figdim)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    #plt.plot(x1, y1, color=clr, label='polynomial fit')
    #plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(r"SFR vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z',# {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(r"SFR vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z',# {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfr_vs_mstar_counts_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=figdim)
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    #plt.plot(x1, y1, color=clr, label='polynomial fit')
    #plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z',# {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z',# {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfr_vs_mstar_vs_iqr_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def plot_sfrsd_vs_mass_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Generates a plot of sfr vs mass with each bin color-coded by median ne
    :param sample_mask: Optional extra mask that is placed after snr cuts
    :return:
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfrsd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        clr = 'r'

    mass_sample = mass[sample_mask]
    sfrsd_sample = sfrsd[sample_mask]
    ne_sample = ne[sample_mask]

    # Define the number of bins
    x_bins = 80
    y_bins = 40

    # font size for labels
    fs = 20

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfrsd_vs_mstar_vs_ne_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfrsd_vs_mstar_counts_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/sfrsd_vs_mstar_vs_iqr_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()



def plot_mass_sfr_sfrsd_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Plot mass, sfr, sfrsd vs ne with percentile trendlines
    :return: none
    """
    mass_bgs = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_bgs = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd_bgs = CC.catalog['SFR_SD'][BGS_MASK]
    z_bgs = CC.catalog['Z'][BGS_MASK]
    ne_bgs, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (z_bgs < Z50)
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (z_bgs < Z90)

    mass = mass_bgs[sample_mask]
    sfr = sfr_bgs[sample_mask]
    sfr_sd = sfr_sd_bgs[sample_mask]
    z = z_bgs[sample_mask]
    ne = ne_bgs[sample_mask]

    fs = 20

    # set percentile line color
    colr = 'dodgerblue'
    colrmap = 'inferno'

    # Plot ne vs mass

    massmin = 8.0
    massmax = 11.5

    # Calculate 25/50/75th percentiles
    ne_75 = []
    ne_50 = []
    ne_25 = []
    mrange = []

    b = 0.1
    for i in np.arange(massmin, massmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(mass >= i, mass < i + b)],
                                          (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            mrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(mass, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(M50, 0, 3.5, color='b')
        #plt.title(f'$n_e$ vs $M_\star$ (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, 0, 3.5, color='r')
        #plt.title(f'$n_e$ vs $M_\star$ (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(f'$n_e$ vs $M_\star$ ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(mrange, ne_25, color='white', linewidth=3.5)
    plt.plot(mrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(mrange, ne_50, color='white', linewidth=3.5)
    plt.plot(mrange, ne_50, color=colr)
    plt.plot(mrange, ne_75, color='white',  linewidth=3.5)
    plt.plot(mrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(massmin, massmax)
    plt.ylim(1, 3)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_ne_vs_mass_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Plot ne vs sfr

    sfrmin = -1.5
    sfrmax = 2.0

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrrange = []

    for i in np.arange(sfrmin, sfrmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfr >= i, sfr < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfr, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(f'$n_e$ vs SFR (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(f'$n_e$ vs SFR (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(f'$n_e$ vs SFR ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_50, color=colr)
    plt.plot(sfrrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrmin, sfrmax)
    plt.ylim(1, 3)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_ne_vs_sfr_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


    # Plot ne vs sfr_sd

    # Change mask to full mass and sfr cuts for this plot so we are only plotting the complete region
    if sample == 2:
        sample_mask = LO_Z_MASK
    elif sample == 3:
        sample_mask = HI_Z_MASK

    sfr_sd = sfr_sd_bgs[sample_mask]
    ne = ne_bgs[sample_mask]

    sfrsdmin = -2
    sfrsdmax = -.25

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrsdrange = []

    b = b/2

    for i in np.arange(sfrsdmin, sfrsdmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfr_sd >= i, sfr_sd < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrsdrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfr_sd, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        #plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrsdrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrsdrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_50, color=colr)
    plt.plot(sfrsdrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrsdmin, sfrsdmax)
    plt.ylim(1, 3)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_ne_vs_sfrsd_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def plot_bpt_ne_color(sample_mask=BGS_SNR_MASK):
    """
    Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)
    Color-codes by n_e
    :return: None
    """
    # potentially change this so instead of a flat snr cut we keep uncertainties
    # and find other ways to deal with it
    snr_lim = SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut(line=NE_LINE_SOURCE)

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(sample_mask, nii_snr > snr_lim, oiii_snr > snr_lim)

    nh = np.log10(nii[bpt_mask] / ha[bpt_mask])  # x-axis
    oh = np.log10(oiii[bpt_mask] / hb[bpt_mask]) # y-axis
    ne = ne[bpt_mask]

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_object_mask         = (oh < agn_boundary(nh)) & (oh < hii_boundary(nh))         # below both red and black lines
    agn_object_mask         = (oh > agn_boundary(nh)) & (oh > shock_boundary(nh))       # above both red and blue
    composite_object_mask   = (oh > hii_boundary(nh)) & (oh < agn_boundary(nh))         # above black and below red
    shock_object_mask       = (oh > agn_boundary(nh)) & (oh < shock_boundary(nh))       # above red and below blue

    hii_ne_median = np.median(ne[hii_object_mask])
    agn_ne_median = np.median(ne[agn_object_mask])
    composite_ne_median = np.median(ne[composite_object_mask])
    shock_ne_median = np.median(ne[shock_object_mask])

    # Arrays to plot the separation lines
    x_for_line_1 = np.log10(np.logspace(-5,.049,300))
    hii_agn_line = hii_boundary(x_for_line_1)           # black dashed
    x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
    composite_line_2 = agn_boundary(x_for_line_2)       # red dotted
    x_for_line_3 = np.linspace(-.13,2,100)
    agn_line_3 = shock_boundary(x_for_line_3)           # blue dotdash

    # Define the number of bins
    x_bins = 70
    y_bins = 60

    # font size for labels
    fs = 18

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        nh, oh, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Creating color map for opaque colorbar
    #cmap = cm.plasma
    cmap = pink_blue_2val_cmap
    #norm = Normalize(vmin=1.5, vmax=2.5)

    fig = plt.figure(figsize=(7, 7))

    # Outer GridSpec: 2 rows (main block + colorbar)
    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[20, 0.5],  # main block tall, colorbar short
        hspace=0.25  # space between them
    )

    # Inner GridSpec: main plot + histograms, tightly packed
    gs_inner = GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs_outer[0],
        hspace=0.0, wspace=0.0  # no gap between top/main/right
    )

    # Axes inside the inner grid
    ax_xDist = fig.add_subplot(gs_inner[0, :3])  # top histogram
    ax_main = fig.add_subplot(gs_inner[1:4, :3])  # main 2D histogram
    ax_yDist = fig.add_subplot(gs_inner[1:4, 3], sharey=ax_main)  # right histogram

    # Colorbar axis in the outer grid, below everything
    ax_cbar = fig.add_subplot(gs_outer[1, 0])

    # Main figure with color-coded ne
    ax_main.set_facecolor('gray')
    X, Y = np.meshgrid(x_edges, y_edges)
    h = ax_main.pcolormesh(X, Y, stat.T, cmap=cmap, shading='auto', vmin=1.824, vmax=2.224)
    # BPT region lines
    ax_main.plot(x_for_line_1, hii_agn_line, color='w', linewidth=3.5)
    ax_main.plot(x_for_line_2, composite_line_2, color='w', linewidth=3.5)
    ax_main.plot(x_for_line_3, agn_line_3, color='w', linewidth=3.5)
    ax_main.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k', linewidth=2.5)
    ax_main.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r', linewidth=2.5)
    ax_main.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b', linewidth=2.5)

    # Text without median values
    ax_main.text(-1.1, -0.4, f"H II", fontweight='bold', fontsize=fs-4)
    ax_main.text(-.22, -0.75, f"Composite", fontweight='bold', fontsize=fs-4)
    ax_main.text(-1.0, 1.1, f"AGN", fontweight='bold', fontsize=fs-4)
    ax_main.text(0.11, -0.25, f"Shocks", fontweight='bold', fontsize=fs-4)
    #plt.text(0.005, 1.005, f'total: {sum(bpt_mask)}, snr $>$ {snr_lim}',
    #      horizontalalignment='left',
    #      verticalalignment='bottom',
    #      transform=ax.transAxes)

    ax_main.set(
        xlim=(-1.25, 0.4),
        ylim=(-1, 1.5)
    )
    ax_xDist.set_xlim(-1.25, 0.4)
    ax_yDist.set_ylim(-1, 1.5)
    ax_main.set_xlabel(
        r'$\log([N II]_{\lambda 6584} / H\alpha)$',
        fontsize=fs
    )
    ax_main.set_ylabel(
        r'$\log([O III]_{\lambda 5007} / H\beta)$',
        fontsize=fs
    )

    lw = 2
    bin_ct = 50

    # Marginal histograms
    ax_yDist.hist(oh[hii_object_mask], bins=50, orientation='horizontal', align='mid', color='b', alpha=0.3)#, histtype='step')
    ax_yDist.hist(oh[composite_object_mask], bins=50, orientation='horizontal', align='mid', color='g', alpha=0.3)#, histtype='step')
    ax_yDist.hist(oh[agn_object_mask], bins=50, orientation='horizontal', align='mid', color='r', alpha=0.3)#, histtype='step')
    ax_xDist.hist(nh[hii_object_mask], bins=50, orientation='vertical', align='mid', color='b', alpha=0.3)#, histtype='step')
    ax_xDist.hist(nh[composite_object_mask], bins=50, orientation='vertical', align='mid', color='g', alpha=0.3)#, histtype='step')
    ax_xDist.hist(nh[agn_object_mask], bins=50, orientation='vertical', align='mid', color='r', alpha=0.3)#, histtype='step')
    """

    from utility_scripts import plot_hist_as_line

    # Marginal line histograms
    # Horizontal histograms → lines
    plot_hist_as_line(ax_yDist, oh[hii_object_mask], bins=50,
                      linestyle=':', orientation='horizontal', alpha=0.5)
    plot_hist_as_line(ax_yDist, oh[composite_object_mask], bins=50,
                      linestyle='--', orientation='horizontal', alpha=0.5)
    plot_hist_as_line(ax_yDist, oh[agn_object_mask], bins=50,
                      linestyle='-', orientation='horizontal', alpha=0.5)
    # Vertical histograms → lines
    plot_hist_as_line(ax_xDist, nh[hii_object_mask], bins=50,
                      linestyle=':', alpha=0.5)
    plot_hist_as_line(ax_xDist, nh[composite_object_mask], bins=50,
                      linestyle='--', alpha=0.5)
    plot_hist_as_line(ax_xDist, nh[agn_object_mask], bins=50,
                      linestyle='-', alpha=0.5)
    """
    # Right histogram settings
    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()
    ax_yDist.yaxis.set_label_position("right")

    # Top histogram settings
    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    yticks = ax_xDist.get_yticks()
    yticklabels = ax_xDist.get_yticklabels()

    new_labels = []
    for t, lab in zip(yticks, yticklabels):
        if t == 0:
            new_labels.append("")  # remove the '0'
        else:
            new_labels.append(lab.get_text())

    ax_xDist.set_yticklabels(new_labels)

    # Colorbar below everything
    cbar = fig.colorbar(h, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'median $\log{n_e/cm^3}$', fontsize=fs-2)

    # Create a tiny inset axes in the blank upper-right corner
    # Coordinates are in figure fraction: (left, bottom, width, height)
    # [left, bottom, width, height]
    leg_ax = fig.add_axes([0.73, 0.68, 0.18, 0.18])
    leg_ax.axis("off")  # hide the box

    # Dummy handles for legend
    h_p = mpl.patches.Patch(color='b', alpha=0.3, label='HII')
    c_p = mpl.patches.Patch(color='g', alpha=0.3, label='COM')
    a_p = mpl.patches.Patch(color='r', alpha=0.3, label='AGN')

    leg_ax.legend(
        handles=[h_p, c_p, a_p],
        loc="upper left",
        frameon=False,
        fontsize=fs-6,
    )

    sam_title = ""
    if sample == 2:
        ax_main.text(0.01, 0.98, f'low-z',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax_main.transAxes, fontsize=fs - 4)
        sam_title = "hi_z"
    elif sample == 3:
        ax_main.text(0.01, 0.98, f'all-z',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax_main.transAxes, fontsize=fs - 4)
        sam_title = "all-z"

    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_bpt_ne_color_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()




    """
    # Perform K-S tests and make cumulative distributions
    # This is now being done in the bpt_ks_tests() function

    out = ks_2samp(np.array(ne[hii_object_mask]), np.array(ne[agn_object_mask]))
    print(out.pvalue)

    plt.hist(ne[hii_object_mask], density=True, cumulative=True, bins=50)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.title("BPT HII galaxies cumulative distribution " + sam_title)
    plt.xlim(1, 3)
    plt.show()

    plt.hist(ne[agn_object_mask], density=True, cumulative=True, bins=50)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.title("BPT AGN galaxies cumulative distribution " + sam_title)
    plt.xlim(1, 3)
    plt.show()
    """



def bpt_ks_tests():
    snr_lim = 3#SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut(line=NE_LINE_SOURCE)

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(nii_snr > snr_lim, oiii_snr > snr_lim)

    loz_mask = generate_combined_mask(LO_Z_MASK, bpt_mask)
    hiz_mask = generate_combined_mask(HI_Z_MASK, bpt_mask)

    nh_lo = np.log10(nii[loz_mask] / ha[loz_mask])  # x-axis
    oh_lo = np.log10(oiii[loz_mask] / hb[loz_mask]) # y-axis
    ne_lo = ne[loz_mask]
    print(len(ne_lo))

    nh_hi = np.log10(nii[hiz_mask] / ha[hiz_mask])  # x-axis
    oh_hi = np.log10(oiii[hiz_mask] / hb[hiz_mask]) # y-axis
    ne_hi = ne[hiz_mask]

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_lo_object_mask         = (oh_lo < agn_boundary(nh_lo)) & (oh_lo < hii_boundary(nh_lo))         # below both red and black lines
    agn_lo_object_mask         = (oh_lo > agn_boundary(nh_lo)) & (oh_lo > shock_boundary(nh_lo))       # above both red and blue
    composite_lo_object_mask   = (oh_lo > hii_boundary(nh_lo)) & (oh_lo < agn_boundary(nh_lo))         # above black and below red
    shock_lo_object_mask       = (oh_lo > agn_boundary(nh_lo)) & (oh_lo < shock_boundary(nh_lo))       # above red and below blue

    hii_hi_object_mask         = (oh_hi < agn_boundary(nh_hi)) & (oh_hi < hii_boundary(nh_hi))         # below both red and black lines
    agn_hi_object_mask         = (oh_hi > agn_boundary(nh_hi)) & (oh_hi > shock_boundary(nh_hi))       # above both red and blue
    composite_hi_object_mask   = (oh_hi > hii_boundary(nh_hi)) & (oh_hi < agn_boundary(nh_hi))         # above black and below red
    shock_hi_object_mask       = (oh_hi > agn_boundary(nh_hi)) & (oh_hi < shock_boundary(nh_hi))       # above red and below blue

    # K-S tests

    print("Comparing low-z sub-samples...")
    hii_agn_lo = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask]))
    print("AGN vs. HII, lo-z:", hii_agn_lo.pvalue)
    hii_com_lo = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[composite_lo_object_mask]))
    print("HII vs. COM, lo-z:", hii_com_lo.pvalue)
    com_agn_lo = ks_2samp(np.array(ne_lo[composite_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask]))
    print("COM vs. AGN, lo-z:", com_agn_lo.pvalue)

    print("Comparing high-z sub-samples...")
    hii_agn_hi = ks_2samp(np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("AGN vs. HII, hi-z:", hii_agn_hi.pvalue)
    hii_com_hi = ks_2samp(np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[composite_hi_object_mask]))
    print("HII vs. COM, hi-z:", hii_com_hi.pvalue)
    com_agn_hi = ks_2samp(np.array(ne_hi[composite_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("COM vs. AGN, hi-z:", com_agn_hi.pvalue)

    print("Comparing low-z vs high-z samples...")
    hii_lo_hi = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_hi[hii_hi_object_mask]))
    print("HII objects high vs low:", hii_lo_hi.pvalue)
    com_lo_hi = ks_2samp(np.array(ne_lo[composite_lo_object_mask]), np.array(ne_hi[composite_hi_object_mask]))
    print("COM objects high vs low:", com_lo_hi.pvalue)
    agn_lo_hi = ks_2samp(np.array(ne_lo[agn_lo_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("AGN objects high vs low:", agn_lo_hi.pvalue)

    fs = 16

    lo_bins = [np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[composite_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask])]
    #clrs_lo = ["#a6cee3", "#1f78b4", "#08306b"]
    clrs_lo = [plt.cm.Blues(i) for i in np.linspace(0.4, 0.9, 3)]

    hi_bins = [np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[composite_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask])]
    #clrs_hi = ["#fb9a99", "#e31a1c", "#67000d"]
    clrs_hi = [plt.cm.Reds(i) for i in np.linspace(0.4, 0.9, 3)]

    bin_name = ['HII', 'Composite', 'AGN']
    bins = np.linspace(1, 3, 40)

    fig, ax = plt.subplots()

    for ne_lo, clr_lo, ne_hi, clr_hi, lab in zip(lo_bins, clrs_lo, hi_bins, clrs_hi, bin_name):
        counts, edges = np.histogram(ne_lo, bins=bins)
        cdf = np.cumsum(counts) / np.sum(counts)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centers, cdf, marker='o', mfc='none', label=lab + ' (low)', color=clr_lo)
        counts, edges = np.histogram(ne_hi, bins=bins)
        cdf = np.cumsum(counts) / np.sum(counts)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centers, cdf, marker='o', mfc='none', label=lab + ' (all)', color=clr_hi)
    ks_string = f"K-S test\nHII: \tp = {hii_lo_hi.pvalue:.3e}\nCOM: \tp = {com_lo_hi.pvalue:.3f}\nAGN: \tp = {agn_lo_hi.pvalue:.3f}"
    plt.text(0.02, 0.98, ks_string, ha='left', va='top', transform=ax.transAxes, fontsize=fs-2)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.legend(loc='lower right')
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/bpt_ks.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def bpt_ks_test_pt_2():
    snr_lim = 3#SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut(line=NE_LINE_SOURCE)

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(nii_snr > snr_lim, oiii_snr > snr_lim)

    loz_mask = generate_combined_mask(LO_Z_MASK, bpt_mask)
    hiz_mask = generate_combined_mask(HI_Z_MASK, bpt_mask)

    nh_lo = np.log10(nii[loz_mask] / ha[loz_mask])  # x-axis
    oh_lo = np.log10(oiii[loz_mask] / hb[loz_mask]) # y-axis
    ne_lo = ne[loz_mask]

    nh_hi = np.log10(nii[hiz_mask] / ha[hiz_mask])  # x-axis
    oh_hi = np.log10(oiii[hiz_mask] / hb[hiz_mask]) # y-axis
    ne_hi = ne[hiz_mask]

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_lo_object_mask         = (oh_lo < agn_boundary(nh_lo)) & (oh_lo < hii_boundary(nh_lo))         # below both red and black lines
    hii_hi_object_mask         = (oh_hi < agn_boundary(nh_hi)) & (oh_hi < hii_boundary(nh_hi))         # below both red and black lines

    fs = 16
    bins = np.linspace(1, 3, 40)
    clrs_lo = [plt.cm.Blues(i) for i in np.linspace(0.4, 0.9, 2)]
    clrs_hi = [plt.cm.Reds(i) for i in np.linspace(0.4, 0.9, 2)]

    # Split HII galaxies in half and compare ne - low-z
    median_hii_nh_lo = np.median(nh_lo[hii_lo_object_mask])
    # Select only the hii objects for the n/h axis
    nh_hii_objects = nh_lo[hii_lo_object_mask]
    # Select only the hii objects for the electron density
    ne_lo_hii = ne_lo[hii_lo_object_mask]
    # Split the electron density in half by the n/h axis
    ne_lo_lhs = ne_lo_hii[nh_hii_objects < median_hii_nh_lo]
    ne_lo_rhs = ne_lo_hii[nh_hii_objects >= median_hii_nh_lo]
    ne_ks_test_lo = ks_2samp(np.array(ne_lo_lhs), np.array(ne_lo_rhs))

    fig, ax = plt.subplots()
    counts, edges = np.histogram(ne_lo_lhs, bins=bins)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label=r'low-z, low NII/H$\alpha$', color=clrs_lo[0])
    counts, edges = np.histogram(ne_lo_rhs, bins=bins)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label=r'low-z, high NII/H$\alpha$', color=clrs_lo[1])
    # Don't show the plot yet, we're going to add the other set too

    # Split HII galaxies in half and compare ne - hi-z
    median_hii_nh_hi = np.median(nh_hi[hii_hi_object_mask])
    # Select only the hii objects for the n/h axis
    nh_hii_objects = nh_hi[hii_hi_object_mask]
    # Select only the hii objects for the electron density
    #print(len(hii_lo_object_mask))
    #print(len(ne_lo))
    ne_hi_hii = ne_hi[hii_hi_object_mask]
    # Split the electron density in half by the n/h axis
    ne_hi_lhs = ne_hi_hii[nh_hii_objects < median_hii_nh_hi]
    ne_hi_rhs = ne_hi_hii[nh_hii_objects >= median_hii_nh_hi]
    ne_ks_test_hi = ks_2samp(np.array(ne_hi_lhs), np.array(ne_hi_rhs))

    counts, edges = np.histogram(ne_hi_lhs, bins=bins)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label=r'all-z, low NII/H$\alpha$', color=clrs_hi[0])
    counts, edges = np.histogram(ne_hi_rhs, bins=bins)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label=r'all-z, high NII/H$\alpha$', color=clrs_hi[1])
    plt.legend(loc='lower right')
    ks_string = f"K-S test:\nlow-z:\tp = {ne_ks_test_lo.pvalue:.3e} \nall-z:\t p = {ne_ks_test_hi.pvalue:.3e}"
    plt.text(0.02, 0.98, ks_string, ha='left', va='top', transform=ax.transAxes, fontsize=fs - 2)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.legend(loc='lower right')
    plt.show()





def plot_ne_vs_sfrsd_binned(sample_mask=BGS_SNR_MASK):
    """
    This function bins the sfrsd/mass/ne data and performs O(2) fits to each mass bin.
    It is not a single fit to all the data
    For the more rigorous and complete fit, view the sfrsd_fitting.py file
    This is kept for records and should not be used
    """
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    sfr_sd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    fs = 18

    #mpl_color_wheel = ['#E40303', '#FF8C00', '#FFED00', '#008026', '#004CFF', '#732982']
    mpl_color_wheel = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    mgap = .5
    masses = np.arange(8.5, 11.5, mgap)
    xmin = 1
    xmax = 3

    fig = plt.figure(figsize=(5,5))

    mind = -1
    for m in masses:
        mind += 1
        mcol = mpl_color_wheel[mind]
        mass_bin = (mass > m) & (mass <= m + mgap)
        sfr_sd_binned = sfr_sd[mass_bin]
        ne_binned = ne[mass_bin]

        sfrsd_min = -2.5
        sfrsd_max = 0

        # Calculate 25/50/75th percentiles
        ne_50 = []
        err_50 = []
        sfrsd_range = []

        ne_50_small = []
        sfrsd_range_small = []

        b = 0.25
        for i in np.arange(sfrsd_min, sfrsd_max, b):
            try:
                ne_double_binned = ne_binned[generate_combined_mask(sfr_sd_binned >= i, sfr_sd_binned < i + b)]
                if len(ne_double_binned) < 1:
                    pass
                elif len(ne_double_binned) < 10:
                    # We want there to be at least 10 objects in each bin.
                    p50 = np.median(ne_double_binned)
                    ne_50_small.append(p50)
                    sfrsd_range_small.append(i + b * 0.5)

                else:
                    p50 = np.median(ne_double_binned)
                    e50 = np.std(ne_double_binned) / np.sqrt(len(np.array(ne_double_binned)))

                    #print(p50, e50)
                    # If there is only one object in the bin, its error is 0
                    #if e50 == 0:
                    #    e50 = 1
                    ne_50.append(p50)
                    err_50.append(e50)
                    sfrsd_range.append(i + b * 0.5)
            except IndexError:
                pass

        ne_50 = np.array(ne_50)
        err_50 = np.array(err_50)
        sfrsd_range = np.array(sfrsd_range)
        ne_50_small = np.array(ne_50_small)
        sfrsd_range_small = np.array(sfrsd_range_small)
        # Use inverse of variance as weights (i.e., 1/σ²)
        weights = 1 / err_50 ** 2

        try:
            # Weighted fit
            coeffs = np.polyfit(sfrsd_range, ne_50, deg=2, w=weights)

            # Evaluate the polynomial
            p = np.poly1d(coeffs)

            # Scatter with error bars. Add tiny offset to x-axis just to make plot more readable
            plt.errorbar(sfrsd_range + mind/100, ne_50, color=mcol, ecolor=mcol, yerr=err_50, fmt='o', capsize=5)
            plt.scatter(sfrsd_range_small + mind/100, ne_50_small, facecolors='none', edgecolors=mcol, marker='o')

            # Plot the fit line
            x_fit = np.linspace(min(np.concatenate([sfrsd_range, sfrsd_range_small])), max(np.concatenate([sfrsd_range, sfrsd_range_small])), 500)
            plt.plot(x_fit + mind/100, p(x_fit), color=mcol, label=fr'${m} < \log m \leq {m + mgap}$ ({sum(np.array(mass_bin))})')
        except TypeError:
            pass

    plt.xlabel('$\log{\Sigma_{SFR} / M_\odot / yr / kpc}$', fontsize=fs)
    plt.ylabel('$\log{n_e / cm^3}$', fontsize=fs)
    #plt.title('2nd order polynomial fit to $\Sigma_{SFR}$ vs $n_e$ binned by mass' + f' ({sum(sample_mask)} galaxies)', fontsize=16)
    if sample == 2:
        plt.title("low-z")
    if sample == 3:
        plt.title("all-z")
    plt.ylim(1.5, 3)
    plt.legend(fontsize=fs-8)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_sfrsd_ne_binned_fits_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()


def compare_metallicity(sample_mask=BGS_SNR_MASK):
    o3n2_metallicity = CC.catalog['METALLICITY_O3N2'][BGS_MASK]
    r23_metallicity = CC.catalog['METALLICITY_R23'][BGS_MASK]

    plt.plot(r23_metallicity[sample_mask], o3n2_metallicity[sample_mask], 'o', alpha=0.05)
    plt.ylabel("Z(O3N2)")
    plt.xlabel("Z(R23)")
    plt.ylim(8.2, 9.3)
    plt.xlim(8.2, 9.3)
    plt.show()


#def plot_metallicity_distribution(sample_mask=BGS_SNR_MASK):



def metallicity(sample_mask=BGS_SNR_MASK):
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
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)  # these are both bgs length
    redshift = CC.catalog['Z'][BGS_MASK]

    sample = 0
    tit = "custom sample"
    mlim = 0
    clr = 'k'
    mcenter = 10
    if sample_mask is BGS_SNR_MASK:
        sample = 1
        tit = "all galaxies"
        #mlim = [M50, M90]
        #clr = ['b', 'r']
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (sfr > SFR50) & (mass > M50)
        mlim = M50
        sfrlim = SFR50
        zlim = Z50
        clr = 'b'
        tit = 'low-z'
        mcenter = 9.83454461143  # This is the median stellar mass in the bin
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (sfr > SFR90) & (mass > M90)
        mlim = M90
        sfrlim = SFR90
        zlim = Z90
        clr = 'r'
        tit = 'all-z'
        mcenter = 10.19955620506

    oiii_5007_snr = oiii_5007_flux * oiii_5007_err_inv
    nii_6584_snr = nii_6584_flux * nii_6584_err_inv
    halpha_snr = halpha_flux * halpha_flux_err_inv
    hbeta_snr = hbeta_flux * hbeta_flux_err_inv

    snr_lim = 3
    fs = 18

    # This is just the metallicity lines
    metallicity_mask = generate_combined_mask(oiii_5007_snr > snr_lim, nii_6584_snr > snr_lim, halpha_snr > snr_lim, hbeta_snr > snr_lim)

    # 03N2 from Pettini & Pagel 2004
    O3N2 = np.log10( (oiii_5007_flux / hbeta_flux) / (nii_6584_flux / halpha_flux) )

    # From PP04
    o3n2_metallicity = 8.73 - 0.32 * O3N2

    hii_galaxy_mask, agn_galaxy_mask, _, _ = get_galaxy_type_mask()

    # 4 is arbitrary, we are just removing the galaxies with failed mass fits
    mass_z_mask = generate_combined_mask(metallicity_mask, sfr > sfrlim, mass > 4, BGS_SNR_MASK, ~agn_galaxy_mask)

    # Mass-metallicity relation
    plt.hist2d(mass[mass_z_mask], o3n2_metallicity[mass_z_mask], bins=(120, 90), norm=mpl.colors.LogNorm())
    plt.vlines(mlim, 0, 20, color=clr, label='Lower mass limit')
    plt.xlim(8, 11.5)
    plt.ylim(8, 9)
    plt.colorbar()
    plt.xlabel(r'$\log{M_\star/M_\odot}$')
    plt.ylabel(r'$12 + \log{O/H}$')
    plt.title(tit)
    plt.legend()
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/mass-metallicity_relation_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Keep all galaxies except for AGN, make note that most composite galaxies are in the high-mass region
    # sample_mask includes mass and sfr cuts for the sample in question
    full_mask = generate_combined_mask(metallicity_mask, sample_mask, ~agn_galaxy_mask)

    spearcorr = spearmanr(o3n2_metallicity[full_mask], ne[full_mask])
    plt.hist2d(o3n2_metallicity[full_mask], ne[full_mask], bins=(50, 50), norm=mpl.colors.LogNorm())
    plt.xlim(8.0, 9)
    plt.ylim(1, 3)
    plt.colorbar(label='count')
    plt.xlabel(r'$12 + \log{O/H}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.text(0.02, 0.98, f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}',
             transform=plt.gca().transAxes, fontsize=fs - 6, va='top', ha='left', bbox=dict(
                    facecolor='white',
                    alpha=0.5,
                    edgecolor='none',
                    boxstyle="round,pad=0.3,rounding_size=.3"))
    plt.title(tit)
    if PLOT_SAVE:
        plt.savefig(f"paper_figures/metallicity_ne_{sample}.{FILE_TYPE}", dpi=PLOT_DPI)
    plt.show()

    # --- Figure and grid layout ---
    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 8),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'right': 0.86}  # leave space on right for colorbar
    )

    # --- Low-mass bin ---
    lowmass_metallicity_bin = generate_combined_mask(
        metallicity_mask, sample_mask, ~agn_galaxy_mask, mass > 4, mass < mcenter
    )

    h1 = axes[0].hist2d(
        o3n2_metallicity[lowmass_metallicity_bin],
        ne[lowmass_metallicity_bin],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )
    spearcorr = spearmanr(o3n2_metallicity[lowmass_metallicity_bin], ne[lowmass_metallicity_bin])
    axes[0].set_xlim(8.0, 9)
    axes[0].set_ylim(1, 3)
    axes[0].set_ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    #axes[0].tick_params(labelbottom=True)  # show x ticks but not label
    axes[0].set_xlabel("")  # no x label
    axes[0].hlines(np.median(ne[lowmass_metallicity_bin]), 1, 10, color='k', label='Median $n_e$')
    #axes[0].legend()
    axes[0].text(
        0.06, 1.01,
        tit,
        fontsize=fs, ha='center', va='bottom',
        transform=axes[0].transAxes
    )
    # Add internal title text
    axes[0].text(
        8.02, 2.93,
        f"$\log{{M_\\star}} < {mcenter:.1f}$",
        fontsize=fs-2, ha='left', va='top',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            edgecolor='none',
            boxstyle="round,pad=0.3,rounding_size=.3")
    )
    axes[0].text(
        8.02, 2.76,
        f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}',
        fontsize=fs - 6, va='top', ha='left',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            edgecolor='none',
            boxstyle="round,pad=0.3,rounding_size=.3")
    )

    # --- High-mass bin ---
    highmass_metallicity_bin = generate_combined_mask(
        metallicity_mask, sample_mask, ~agn_galaxy_mask, mass > 4, mass >= mcenter
    )

    h2 = axes[1].hist2d(
        o3n2_metallicity[highmass_metallicity_bin],
        ne[highmass_metallicity_bin],
        bins=(50, 50),
        norm=mpl.colors.LogNorm()
    )

    spearcorr = spearmanr(o3n2_metallicity[highmass_metallicity_bin], ne[highmass_metallicity_bin])
    axes[1].set_xlim(8.0, 9)
    axes[1].set_ylim(1, 3)
    axes[1].set_xlabel(r'$12 + \log{O/H}$', fontsize=fs)
    axes[1].set_ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    axes[1].hlines(np.median(ne[highmass_metallicity_bin]), 1, 10, color='k', label='Median $n_e$')
    axes[1].legend(loc="lower left", fontsize=fs-2)

    axes[1].text(
        8.02, 2.93,
        f"$\log{{M_\\star}} \geq {mcenter:.1f}$",
        fontsize=fs-2, ha='left', va='top',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            edgecolor='none',
            boxstyle="round,pad=0.3,rounding_size=.3")
    )
    axes[1].text(
        8.02, 2.76,
        f'spearman statistic: {spearcorr.statistic:.2f}\np-value: {spearcorr.pvalue:.3e}',
        fontsize=fs - 6, va='top', ha='left',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            edgecolor='none',
            boxstyle="round,pad=0.3,rounding_size=.3")
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
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_metallicity_ne_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

def total_sfr_sd(sample_mask = BGS_SNR_MASK):
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    # Half-light radius
    radius = CC.catalog['SHAPE_R'][BGS_MASK]
    #print(sum(radius[BGS_SNR_MASK] <= 0))

    sfrsd = sfr / (np.pi * radius ** 2)
    ne, _ = bgs_ne_snr_cut(line=NE_LINE_SOURCE)
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    mlim = 0
    clr = 'k'
    redshift_sample_mask = sample_mask
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50) & (radius > 0)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90) & (radius > 0)
        mlim = M90
        clr = 'r'

    #mass_sample = mass[sample_mask]
    #sfrsd_sample = sfrsd[sample_mask]
    #ne_sample = ne[sample_mask]

    # Define the number of bins
    x_bins = 90
    y_bins = 75

    # font size for labels
    fs = 20
    ymin = -2.5
    ymax = 3

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='median',
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_vs_ne_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    """
    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='count',
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_counts_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr,
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)  # , norm=mpl.colors.LogNorm())
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_vs_iqr_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()
    """


    # Plot ne vs sfr_sd

    # If we include all completeness cuts, all of the SFRSD < 0 disappear and I cannot figure out why
    trendline_mask = np.array(redshift_sample_mask & (mass >= mlim))

    sfrsd = sfrsd[trendline_mask]
    mass = mass[trendline_mask]
    ne = ne[trendline_mask]

    print(min(sfrsd))

    sfrsdmin = -0.1
    sfrsdmax = 0.7

    colrmap = 'inferno'
    colr = 'dodgerblue'

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrsdrange = []

    b = 0.02

    for i in np.arange(sfrsdmin, sfrsdmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfrsd >= i, sfrsd < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrsdrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfrsd, ne, bins=(500,70), cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        #plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrsdrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrsdrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_50, color=colr)
    plt.plot(sfrsdrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrsdmin, sfrsdmax)
    plt.ylim(1, 3)
    if PLOT_SAVE:
        plt.savefig(f'paper_figures/paper_ne_vs_sfrsd_total_{sample}.{FILE_TYPE}', dpi=PLOT_DPI)
    plt.show()



def generate_all_plots_for_paper():

    # Plot sfr and mass vs redshift with completeness limits labeled
    plot_redshift_vs_mass_sfr()

    # Plot sfr main sequence
    plot_sfr_ms(sample_mask=LO_Z_MASK)
    plot_sfr_ms(sample_mask=HI_Z_MASK)

    # Plot redshift vs ne
    plot_redshift_vs_ne(sample_mask=LO_Z_MASK)
    plot_redshift_vs_ne(sample_mask=HI_Z_MASK)

    # Plot sfr ms with ne colored bins
    plot_sfr_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfr_vs_mass_vs_ne(sample_mask=HI_Z_MASK)

    # Plot sfrsd vs mass with ne colored bins
    plot_sfrsd_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfrsd_vs_mass_vs_ne(sample_mask=HI_Z_MASK)

    # Plot ne vs mass, sfr, sfrsd with percentile trendlines
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=LO_Z_MASK)
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=HI_Z_MASK)

    # Plot SFRSD vs ne evolution in different bins
    plot_ne_vs_sfrsd_binned(sample_mask=LO_Z_MASK)
    plot_ne_vs_sfrsd_binned(sample_mask=HI_Z_MASK)

    # Plot BPT diagram color-coded by median ne
    plot_bpt_ne_color(sample_mask=LO_Z_MASK)
    plot_bpt_ne_color(sample_mask=HI_Z_MASK)
    bpt_ks_tests()

    # Plot metallicity properties
    metallicity(sample_mask=LO_Z_MASK)
    metallicity(sample_mask=HI_Z_MASK)

    # Plot SFRSD figures
    total_sfr_sd(sample_mask=LO_Z_MASK)
    total_sfr_sd(sample_mask=HI_Z_MASK)


def generate_plots_for_proposal():
    plot_redshift_vs_mass_sfr()
    plot_sfr_vs_mass_vs_ne(sample_mask=LO_Z_MASK)


def generate_chosen_plots():
    #compare_metallicity(sample_mask=LO_Z_MASK)
    #metallicity(sample_mask=LO_Z_MASK)
    #metallicity(sample_mask=HI_Z_MASK)
    #plot_sfr_ms(sample_mask=LO_Z_MASK)
    #plot_sfr_ms(sample_mask=HI_Z_MASK)
    #plot_ne_distribution(sample_mask=LO_Z_MASK)
    #plot_ne_distribution(sample_mask=HI_Z_MASK)
    #plot_bpt_ne_color(sample_mask=LO_Z_MASK)
    #plot_bpt_ne_color(sample_mask=HI_Z_MASK)
    #bpt_ks_tests()
    #bpt_ks_test_pt_2()
    #compare_sfr(sample_mask=LO_Z_MASK)
    #compare_sfr(sample_mask=HI_Z_MASK)
    #plot_redshift_vs_mass_sfr()
    plot_redshift_vs_ne(sample_mask=LO_Z_MASK)
    plot_redshift_vs_ne(sample_mask=HI_Z_MASK)

    histogram_plots()

    pass




def main():
    global PLOT_DPI
    global PLOT_SAVE
    global FILE_TYPE
    global NE_LINE_SOURCE
    global SNR_LIM
    SNR_LIM = 5
    PLOT_DPI = 300
    PLOT_SAVE = True
    FILE_TYPE = 'png'
    NE_LINE_SOURCE = 0 # 0 for both, 1 for oii, 2 for sii

    #generate_all_plots_for_paper()
    #generate_plots_for_proposal()

    generate_chosen_plots()


if __name__ == '__main__':
    main()