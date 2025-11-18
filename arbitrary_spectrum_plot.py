import os
import mks

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

from utility_scripts import get_lum, generate_combined_mask


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',

    # --- layout & appearance ---
    #'axes.linewidth': 1.2
})

import time
import hashlib
import glob
import random

import pandas as pd

# import DESI related modules -
import desispec.io                             # Input/Output functions related to DESI spectra
from desispec import coaddition                # Functions related to coadding the spectra

import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

# DESI targeting masks -
from desitarget.cmx.cmx_targetmask import cmx_mask as cmxmask
from desitarget.sv1.sv1_targetmask import desi_mask as sv1mask
from desitarget.sv2.sv2_targetmask import desi_mask as sv2mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3mask
from desitarget.targetmask import desi_mask as specialmask


class Spectrum:
    def __init__(self, targetid='random', zpix_cat=None):

        #print("reading in table...")
        self.basedir = os.path.expanduser('~') + '/Documents/school/research'
        self.my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.survey = 'sv3'
        self.program = 'bright'
        self.specprod = 'fuji'
        self.specprod_dir = f'{self.my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.fastspec_dir = f'{self.my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2'
        self.healpix_dir = self.specprod_dir + '/healpix'
        if zpix_cat is None:
            self.zpix_cat = Table.read(f'{self.specprod_dir}/zcatalog/zall-pix-{self.specprod}.fits', hdu="ZCATALOG")
        else:
            self.zpix_cat = zpix_cat
        #self.fsf_models = Table.read(f'{self.fastspec_dir}/fastspec-fuji.fits', hdu=3)

        if targetid == 'random':
            bgs_tgtmask = sv3mask["BGS_ANY"]
            #targetid = random.choice(self.zpix_cat['TARGETID'][self.zpix_cat['SV3_BGS_TARGET'] > 0])
            targetid = random.choice(self.zpix_cat['TARGETID'][(self.zpix_cat["SV3_DESI_TARGET"] & bgs_tgtmask != 0)])# & self.zpix_cat["ZWARN"] > 0)])

            self.targetid = targetid
            print(f"targetid {targetid}")
            self.check_for_files()  # if plotting a random spectrum, we're assuming we need to check if the files exist
        else:
            self.targetid = targetid
            print(f"targetid {targetid}")

        #self.files_checked = False
        #self.check_for_files()

    def plot_spectrum(self, foldstruct="spectra/", display_plot=False):
        """
        return 6 means there were multiple matching targetids found in the selected fastspecfit file

        :param foldstruct:
        :return:
        """

        selected_tgts = self.zpix_cat['TARGETID'] == self.targetid

        zcat_sel = self.zpix_cat[selected_tgts]

        survey_col = zcat_sel['SURVEY'].astype(str)
        program_col = zcat_sel['PROGRAM'].astype(str)
        hpx_col = zcat_sel['HEALPIX']
        redshift_col = zcat_sel['Z']

        is_primary = zcat_sel['ZCAT_PRIMARY']

        for survey, program, hpx, redshift in zip(survey_col[is_primary], program_col[is_primary], hpx_col[is_primary], redshift_col[is_primary]):
            # This loop should only run once, as there should be only one primary spectrum for a given tid
            print(survey, program, hpx, redshift)
            tgt_dir = f'{self.healpix_dir}/{survey}/{program}/{hpx // 100}/{hpx}'
            zFact = redshift + 1

            # Filename
            coadd_filename = f'coadd-{survey}-{program}-{hpx}.fits'
            specfile = f'{tgt_dir}/{coadd_filename}'

            spec = read_spectra(specfile).select(targets=self.targetid)

            coadd_spec = coadd_cameras(spec)
            bands = coadd_spec.bands[0]

            fastfile = self.fastspec_dir + f'/healpix/{survey}/{program}/{hpx // 100}/{hpx}/fastspec-{survey}-{program}-{hpx}.fits.gz'
            meta = Table(fitsio.read(fastfile, 'METADATA'))
            data = Table(fitsio.read(fastfile, 'FASTSPEC'))

            #zwarn is currently never set
            zflag = False

            fsf_target_mask = data['TARGETID'] == self.targetid

            if sum(fsf_target_mask) > 1:
                print("more than one matching targetid was found in the fastspec table. please check the data.")
                return 6
            oii_rat = data['OII_DOUBLET_RATIO'][fsf_target_mask][0]
            oii_snr_26 = data['OII_3726_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OII_3726_AMP_IVAR'][fsf_target_mask][0])
            oii_snr_29 = data['OII_3729_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OII_3729_AMP_IVAR'][fsf_target_mask][0])

            #sii_rat = data['SII_DOUBLET_RATIO'][fsf_target_mask][0]
            #sii_snr_16 = data['SII_6716_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['SII_6716_AMP_IVAR'][fsf_target_mask][0])
            #sii_snr_31 = data['SII_6731_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['SII_6731_AMP_IVAR'][fsf_target_mask][0])

            nii48_snr = data['NII_6548_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['NII_6548_AMP_IVAR'][fsf_target_mask][0])
            ha_snr = data['HALPHA_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['HALPHA_AMP_IVAR'][fsf_target_mask][0])
            nii84_snr = data['NII_6584_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['NII_6584_AMP_IVAR'][fsf_target_mask][0])

            models, hdr = fitsio.read(fastfile, 'MODELS', header=True)
            models = models[meta['TARGETID'] == self.targetid]
            modelwave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1']

            mw_transmission_spec = dust_transmission(coadd_spec.wave[bands], meta['EBV'][meta['TARGETID'] == self.targetid])

            line_names = [r'$[OII]$', r'$[OIII]$', r'$[OIII]$', r'$H\alpha$', r'$H\beta$', r'$H\gamma$',
                          r'$H\delta$', r'$[SII]$', r'$[SII]$', r'$CaII H$', r'$CaII K$', r'$[NII]$',
                          r'$[NII]$', r'$[NeIII]$']
            line_vals = [3728.48, 4960.2937, 5008.2383, 6564.613, 4862.683, 4341.684, 4102.892, 6718.2913, 6732.6705, 3933, 3968, 6549.8578, 6585.2696, 3869.8611]

            #lines = {line_names[i]: line_vals[i] for i in range(len(line_names))}

            spec_lo = 3500
            spec_hi = 7000

            oii_limit = 8
            nii_limit = 25
            #sii_limit = 20

            # this is the wl to use as the center of the subfig
            oii_line = (3727.0919 + 3729.8750) / 2
            #sii_line = (6718.2913 + 6732.6705) / 2
            nii_line = (6549.8578 + 6564.613 + 6585.2696) / 3 # average of the 3 lines showing in that subfig

            # buff determines how much space to put above and below the top of the lines
            buff = 0.15

            full_spec = coadd_spec.flux['brz'][0] / mw_transmission_spec

            # get maxima and minima for full plot
            full_left_lim = coadd_spec.wave['brz'] / zFact > (spec_lo)
            full_right_lim = coadd_spec.wave['brz'] / zFact < (spec_hi)
            full_xlims = np.logical_and(full_left_lim, full_right_lim)
            full_y_top = max(full_spec[full_xlims])
            full_y_bottom = min(full_spec[full_xlims])
            full_y_range = full_y_top - full_y_bottom
            full_y_top += full_y_range*buff
            full_y_bottom -= full_y_range*buff

            # get maxima and minima for oii subplot
            oii_left_lim = coadd_spec.wave['brz'] / zFact > (oii_line-oii_limit)
            oii_right_lim = coadd_spec.wave['brz'] / zFact < (oii_line+oii_limit)
            oii_xlims = np.logical_and(oii_left_lim, oii_right_lim)
            try:
                oii_y_top = max(full_spec[oii_xlims])
                oii_y_bottom = min(full_spec[oii_xlims])
            except ValueError:
                oii_y_top = 1
                oii_y_bottom = 0
            oii_y_range = oii_y_top - oii_y_bottom
            oii_y_top += oii_y_range*buff
            oii_y_bottom -= oii_y_range*buff

            # get maxima and minima for sii subplot
            #sii_left_lim = coadd_spec.wave['brz'] / zFact > (sii_line-sii_limit)
            #sii_right_lim = coadd_spec.wave['brz'] / zFact < (sii_line+sii_limit)
            #sii_xlims = np.logical_and(sii_left_lim, sii_right_lim)
            #try:
            #    sii_y_top = max(full_spec[sii_xlims])
            #    sii_y_bottom = min(full_spec[sii_xlims])
            #except ValueError:
            #    sii_y_top = 1
            #    sii_y_bottom = 0
            #sii_y_range = sii_y_top - sii_y_bottom
            #sii_y_top += sii_y_range*buff
            #sii_y_bottom -= sii_y_range*buff

            # get maxima and minima for nii subplot
            nii_left_lim = coadd_spec.wave['brz'] / zFact > (nii_line-nii_limit)
            nii_right_lim = coadd_spec.wave['brz'] / zFact < (nii_line+nii_limit)
            nii_xlims = np.logical_and(nii_left_lim, nii_right_lim)
            try:
                nii_y_top = max(full_spec[nii_xlims])
                nii_y_bottom = min(full_spec[nii_xlims])
            except ValueError:
                nii_y_top = 1
                nii_y_bottom = 0
            nii_y_range = nii_y_top - nii_y_bottom
            nii_y_top += nii_y_range*buff
            nii_y_bottom -= nii_y_range*buff

            fs = 12

            fig = plt.figure(figsize=(6, 4.5))
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            axes = [ax1, ax2]

            # plotting oii spectrum
            ax1.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4, label='Data')
            #ax1.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax1.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='FSF model', ls='-', color='red', linewidth=1)
            ax1.set_xlim([oii_line-oii_limit, oii_line+oii_limit])
            ax1.set_ylim([oii_y_bottom, oii_y_top])
            #ax1.text(0.995, 0.975, f'[O II]/[O II]\n' + 'ratio: {0:.4f}\n'.format(oii_rat) + 'SNR: {0:.1f}'.format(oii_snr_26) + ', {0:.1f}'.format(oii_snr_29),
            #         horizontalalignment='right',
            #         verticalalignment='top',
            #         transform=ax1.transAxes)

            ax1.axvline(3727.0919, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax1.axvline(3729.8750, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax1.set_ylabel(r'Flux [$10^{-17}$ erg cm$^{2}$ s$^{-1}$ \AA$^{-1}$]', fontsize=fs)
            ax1.set_xlabel(r'$\lambda_{rest}$ [\AA]', fontsize=fs)
            #ax1.text(0.01, 1.01, f'targetid = {self.targetid}', fontsize=fs-3, transform=ax1.transAxes,
            #         horizontalalignment='left', verticalalignment='bottom')


            # plotting nii spectrum
            ax2.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax2.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax2.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), ls='-', color='red', linewidth=1)

            ax2.set_xlim([nii_line-nii_limit, nii_line+nii_limit])
            ax2.set_ylim([nii_y_bottom, nii_y_top])
            #ax2.text(0.995, 0.975, fr'[N II]/H$\alpha$/[N II]' + '\nSNR: {0:.1f}'.format(nii48_snr) + ', {0:.1f}'.format(ha_snr) + ', {0:.1f}'.format(nii84_snr),
            #         horizontalalignment='right',
            #         verticalalignment='top',
            #         transform=ax2.transAxes)
            ax2.axvline(6549.8578, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax2.axvline(6564.613, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax2.axvline(6585.2696, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax2.set_xlabel(r'$\lambda_{rest}$ [\AA]', fontsize=fs)

            # --- combine legends from both axes ---
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

            # Place a single centered legend above both panels, without a box
            plt.legend(handles, labels,
                       loc='lower center',
                       bbox_to_anchor=(-0.09, 1.02),
                       ncol=len(set(labels)),  # optional: arrange horizontally
                       frameon=False,
                       fontsize=fs-3)

            plt.subplots_adjust(bottom=0.13, right=0.97)

            # plotting sii spectrum
            #ax7.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
            #         color='black', alpha=0.4)
            #ax7.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
            #         #color='k', lw=1.0)
            #ax7.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            #ax7.set_xlim([sii_line - sii_limit, sii_line + sii_limit])
            #ax7.set_ylim([sii_y_bottom, sii_y_top])
            #ax7.text(0.995, 0.975, f'[SII]/[SII]\n' + '{0:.4f}\n'.format(sii_rat) + 'SNR: {0:.1f}'.format(sii_snr_16) + ', {0:.1f}'.format(sii_snr_31),
            #         horizontalalignment='right',
            #         verticalalignment='top',
            #         transform=ax7.transAxes)
            #ax7.axvline(6718.2913, linestyle='dashed', lw = 0.8, alpha=0.4)
            #ax7.axvline(6732.6705, linestyle='dashed', lw = 0.8, alpha=0.4)
            #ax7.set_xlabel(r'$\lambda_{rest}$')
            """
            try:
                plt.savefig(f'{foldstruct}spectrum_{self.targetid}.png', dpi=800)
            except FileNotFoundError:
                os.mkdir(fold)
            """
            plt.savefig(f'{foldstruct}limtedspectrum_{self.targetid}.png', dpi=FIG_DPI)
            if display_plot:
                plt.show()

    def check_for_files(self):
        zcat_sel = self.zpix_cat[self.zpix_cat['TARGETID'] == self.targetid]

        survey_col = zcat_sel['SURVEY'].astype(str)
        program_col = zcat_sel['PROGRAM'].astype(str)
        hpx_col = zcat_sel['HEALPIX']

        spectra_path_prefix = self.healpix_dir
        fsf_path_prefix = self.fastspec_dir + f'/healpix'
        local_path_prefix_list = [spectra_path_prefix, fsf_path_prefix]

        spectra_web_path_prefix = f'/public/edr/spectro/redux/{self.specprod}/healpix'
        fsf_web_path_prefix = f'/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/healpix'
        web_path_prefix_list = [spectra_web_path_prefix, fsf_web_path_prefix]

        for local_path_prefix, web_path_prefix in zip(local_path_prefix_list, web_path_prefix_list):
            for survey, program, healpix in zip(survey_col, program_col, hpx_col):
                local_path = local_path_prefix + f'/{survey}/{program}/{healpix // 100}/{healpix}'
                web_path = web_path_prefix + f'/{survey}/{program}/{healpix // 100}/{healpix}'

                try:  # try to get the path to the hash file
                    hashfile_path = glob.glob(local_path + '/*.sha256sum')[0]
                except IndexError:  # if it does not exist, download it and get the path
                    print(f"hash file for {survey}/{program}/{healpix} not found, downloading hash file...")
                    #print(f'wget -q -r -nH --no-parent -e robots=off --reject="index.html*" -A.sha256sum --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/')
                    print(f"downloading data from https://data.desi.lbl.gov{web_path}/")
                    os.system(f'wget -q -r -nH --no-parent -e robots=off --reject="index.html*" -A.sha256sum --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/')
                    hashfile_path = glob.glob(local_path + '/*.sha256sum')[0]
                    print("hash file successfully downloaded.")

                df = pd.read_csv(hashfile_path, sep='\s+', header=None)
                file_names = df[1]
                hashes = df[0]

                for file_name, hash in zip(file_names, hashes):
                    #print(file_name, hash)
                    file_path = local_path + '/' + file_name
                    file_exists = False
                    hash_good = False
                    fail_counter = 0

                    while not file_exists or not hash_good:
                        if fail_counter > 3:
                            print("failed to download and successfully verify file. the file may be corrupted on the server. ending session.")
                            return 1
                        file_exists = os.path.isfile(file_path)
                        if file_exists:
                            print(f"{file_name} exists. verifying...", end=" ")
                            hashed_file = self.hash_file(file_path)
                            hash_good = hashed_file.hexdigest() == hash
                            if hash_good:
                                print("file is good.")
                            if not hash_good:
                                print("the file could not be verified. delete file and redownload? (y/N)")
                                accept = str(input())
                                if accept == 'y' or accept == 'Y':
                                    os.remove(file_path)
                                else:
                                    print("ending session.")
                                    return 1
                        if not file_exists:
                            print(f"{file_name} does not exist on this machine. Downloading file...", end=" ")
                            os.system(f'wget -r -q -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/' + file_name)
                            print("download complete. verifying...", end=" ")
                            hashed_file = self.hash_file(file_path)
                            hash_good = hashed_file.hexdigest() == hash
                            if hash_good:
                                print("file is good.")
                                file_exists = True
                            else:
                                print("the file could not be verified. something may have gone wrong with the download. delete file and try again? (y/N)")
                                accept = str(input())
                                if accept == 'y' or accept == 'Y':
                                    os.remove(file_path)
                                else:
                                    print("ending session.")
                                    return 1
                                print(f"removing {file_path}")
                                fail_counter += 1

        return 0

    def hash_file(self, file_path):
        BUF_SIZE = 65536

        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256

def spec_plot(*tids):

    for tid in tids:
        spec = Spectrum(targetid=tid)
        spec.check_for_files()
        spec.plot_spectrum(display_plot=True)  # this makes the plot and saves it


def main():
    global FIG_DPI
    FIG_DPI = 250

    spec_plot(39627908943188094, 39627908943188039, 39627908943188364)



if __name__ == '__main__':
    main()
