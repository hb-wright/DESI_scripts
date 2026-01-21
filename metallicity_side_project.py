import fitsio
import numpy as np
from astropy.table import Table
from astropy.table import vstack
import astropy.units as u
import os
from astropy.cosmology import FlatLambdaCDM


def read_table():
    ccatDir = os.path.expanduser('~') + '/Documents/school/research/customcatalog'
    try:
        catalog = Table.read(f"{ccatDir}/dr1_metallicity_subset.fits")
    except FileNotFoundError:
        catalog = create_table()

    return catalog


def create_table():
    myDir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'iron'
    dr = 'dr1'
    vn = '2.1'
    #specprod_dir = f'{my_dir}/public/{dr}/spectro/redux/{specprod}/'
    fsfCatalogsDirDR1 = f'{myDir}/public/{dr}/vac/{dr}/fastspecfit/{specprod}/v{vn}/catalogs'
    fsfCatalogsDirEDR = f'{myDir}/public/edr/vac/edr/fastspecfit/fuji/v3.2/catalogs'

    fitsFileNames = []
    fitsFileNames.append(f'{fsfCatalogsDirEDR}/fastspec-fuji.fits')
    fitsFileNames.append(f'{fsfCatalogsDirDR1}/fastspec-iron.fits')

    obsid_dict = {1083435338694661: "ngc5356",
                  39627914962013684: "ngc5356",
                  39627914962013611: "ngc5356",
                  39627914962013513: "ngc5356",
                  2394053199003654: "ngc5356",
                  2402849287831572: "ngc5348",
                  1083435334500385: "ngc5348",
                  39627914957819967: "ngc5348",
                  39627914957819970: "ngc5348",
                  39627914957819959: "ngc5348",
                  1083435334500386: "ngc5348",
                  39627884939187718: "ngc5560",
                  39627872884752915: "ngc5577",
                  2407205261279237: "ngc5577",
                  2847009912389632: "ngc5577",
                  39627872884752549: "ngc5577",
                  39627872884752508: "ngc5577",
                  2390246289309697: "pgc25063",
                  39628296685621402: "ngc6186"}

    # list of target IDs we need
    wanted = set(obsid_dict.keys())
    found = set()  # track IDs found in any file

    needed_cols = [
        "TARGETID", "SURVEY", "PROGRAM", "Z",
        "OII_3726_FLUX", "OII_3726_FLUX_IVAR",
        "OII_3729_FLUX", "OII_3729_FLUX_IVAR",
        "HBETA_FLUX", "HBETA_FLUX_IVAR",
        "OIII_4959_FLUX", "OIII_4959_FLUX_IVAR",
        "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR",
        "HALPHA_FLUX", "HALPHA_FLUX_IVAR",
        "NII_6584_FLUX", "NII_6584_FLUX_IVAR"
    ]

    file_labels = {
        "fastspec-fuji.fits": "fuji",
        "fastspec-iron.fits": "iron"
    }

    tables = []

    for filename in fitsFileNames:
        print("processing", filename.split("/")[-1])
        with fitsio.FITS(filename, 'r') as f:

            hdu1 = f[1]
            hdu2 = f[2]

            target_ids = hdu1.read_column("TARGETID")
            mask = np.isin(target_ids, list(wanted))
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            # Read only the needed columns, matched rows
            arr1 = hdu1.read(columns=needed_cols, rows=idx)
            t = Table(arr1)

            ra = hdu2.read_column("RA", rows=idx)
            dec = hdu2.read_column("DEC", rows=idx)

            t["RA"] = ra
            t["DEC"] = dec

            # Add SOURCE column
            source_label = file_labels.get(filename.split("/")[-1], 'unknown')
            t["RELEASE"] = source_label

            # Add GALAXY column via dict lookup
            t["GALAXY"] = [obsid_dict.get(int(tid), None) for tid in t["TARGETID"]]

            # Update found IDs
            found.update(t["TARGETID"])

            tables.append(t)

    # Combine everything
    final_table = vstack(tables, join_type='outer')

    # IDs not found at all
    missing = wanted - found
    print("Number not found:", len(missing))
    print("Missing TARGETIDs:", missing)

    return final_table


def write_table(catalog):
    for col in catalog.colnames:
        data = catalog[col]

        # Only operate on float columns
        if np.issubdtype(data.dtype, np.floating):
            # Replace inf / -inf with nan
            bad = ~np.isfinite(data)
            if np.any(bad):
                data[bad] = np.nan

    ccatDir = os.path.expanduser('~') + '/Documents/school/research/customcatalog'
    print("saving tables")
    # write your compact table to disk for future use
    catalog.write(f"{ccatDir}/dr1_metallicity_subset.fits", overwrite=True)


def k_lambda_2000(wavelength):
    # Wavelength is in angstroms - convert to microns
    wl = wavelength * 1e-4

    if wl <= 2.2000 and wl > .6300:
        k = 2.659 * (-1.857 + (1.040 / wl)) + 4.05
    elif wl >= .1200:
        k = 2.659 * (-2.156 + (1.509 / wl) - (0.198 / (wl ** 2)) + (0.011 / (wl ** 3))) + 4.05
    else:
        print(wavelength, "outside wavelength range")
        return 0

    return k


def get_lum(f, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    D_l = cosmo.luminosity_distance(z).cgs.value #this puts dL in cm
    f = f * 1E-17 #flux in erg/cm^2 now
    return f*4*np.pi*D_l**2


def balmer_correction(catalog):
    # Calculate extinction-corrected H-alpha using H-beta
    E_beta_alpha = 2.5 * np.log10(2.86 / (catalog['HALPHA_FLUX'] / catalog['HBETA_FLUX']))
    EBV = E_beta_alpha / (k_lambda_2000(6563) - k_lambda_2000(4861))
    EBV_s = EBV * 0.44  # from Calzetti+2000
    A_halpha = k_lambda_2000(6563) * EBV_s
    A_halpha[np.where(A_halpha < 0)] = 0

    # Finally calculate the correction and apply it
    correction = 10 ** (0.4 * A_halpha)
    ha_balmer_corrected = catalog['HALPHA_FLUX'] * correction

    return ha_balmer_corrected, A_halpha


def calculate_sfr(ha, redshifts):

    h_alpha_lum = np.empty(len(ha))
    for i, (flux, z) in enumerate(zip(ha, redshifts)):
        h_alpha_lum[i] = get_lum(flux, z)

    # using the table from Kennicutt 2012
    halpha_sfr_log = np.log10(h_alpha_lum) - 41.27
    # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
    #halpha_sfr = h_alpha_lum * 7.9E-42

    return 10 ** halpha_sfr_log


def calculate_o3n2_metallicity(oiii_5007_flux, nii_6584_flux, halpha_flux, hbeta_flux):

    # 03N2 from Pettini & Pagel 2004
    O3N2 = np.log10((oiii_5007_flux / hbeta_flux) /
                    (nii_6584_flux / halpha_flux))
    o3n2_metallicity = 8.73 - 0.32 * O3N2

    return o3n2_metallicity


def calculate_r23_metallicity(oii_3726_flux, oii_3729_flux, oiii_4959_flux, oiii_5007_flux, hbeta_flux):

    # R23 from Tremonti 2004
    R23 = np.log10((oii_3726_flux + oii_3729_flux + oiii_4959_flux + oiii_5007_flux) / hbeta_flux)
    r23_metallicity = 9.185 - 0.313 * R23 - 0.264 * R23**2 - 0.321 * R23**3

    return r23_metallicity


def add_to_table(catalog):
    print("Calculating values")
    ha_corrected, A_halpha = balmer_correction(catalog)

    catalog['A_HALPHA'] = A_halpha
    catalog['SFR'] = calculate_sfr(catalog['HALPHA_FLUX'], catalog['Z'])
    catalog['SFR_CORRECTED'] = calculate_sfr(ha_corrected, catalog['Z'])
    catalog['METALLICITY_O3N2'] = calculate_o3n2_metallicity(catalog['OIII_5007_FLUX'],
                                                       catalog['NII_6584_FLUX'],
                                                       catalog['HALPHA_FLUX'],
                                                       catalog['HBETA_FLUX'])
    catalog['METALLICITY_R23'] = calculate_r23_metallicity(catalog['OII_3726_FLUX'],
                                                           catalog['OII_3729_FLUX'],
                                                           catalog['OIII_4959_FLUX'],
                                                           catalog['OIII_5007_FLUX'],
                                                           catalog['HBETA_FLUX'])
    original_cols = list(catalog.colnames)
    reordered = (
            ["TARGETID", "GALAXY", "RELEASE", "SURVEY", "PROGRAM", "RA", "DEC", "Z", "A_HALPHA", "SFR", "SFR_CORRECTED", "METALLICITY_O3N2", "METALLICITY_R23"] +
            [c for c in original_cols if c not in ("TARGETID", "GALAXY", "RELEASE", "SURVEY", "PROGRAM", "RA", "DEC", "Z", "A_HALPHA", "SFR", "SFR_CORRECTED", "METALLICITY_O3N2", "METALLICITY_R23")]
    )
    t = catalog[reordered]

    return t


def main():
    catalog = read_table()
    catalog = add_to_table(catalog)
    write_table(catalog)


if __name__ == '__main__':
    main()