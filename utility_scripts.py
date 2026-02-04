import numpy as np
import pandas as pd
import os
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import time

#from sample_masks import BGS_SNR_MASK, BGS_MASK



class CustomTimer:
    def __init__(self, dlen, calcstr=''):
        self.t = time.time()
        self.lastFullSecElapsed = int(time.time() - self.t)
        self.dataLength = int(dlen)
        if calcstr != '':
            self.calcstr = ' ' + str(calcstr)
        else:
            self.calcstr = calcstr
    def update_time(self, i):
        elapsed = time.time() - self.t
        fullSecElapsed = int(elapsed)
        if fullSecElapsed > self.lastFullSecElapsed:
            self.lastFullSecElapsed = fullSecElapsed
            percent = 100 * (i + 1) / self.dataLength
            totalTime = elapsed / (percent / 100)
            remaining = totalTime - elapsed
            trString = (f"Calculating{self.calcstr}, " + str(int(percent)) + "% complete. approx "
                        + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
            print('\r' + trString, end='', flush=True)


def check_files(desi_id, specprod = 'fuji', my_dir = '/Documents/school/research/desidata'):
    """
    input: desi_id is string
    :return: None
    """
    homedir = os.path.expanduser('~')
    my_dir = homedir + my_dir

    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
    short_id = str(desi_id)[:-2]

    file = f'{specprod_dir}/healpix/sv1/bright/{short_id}/{desi_id}/spectra-sv1-bright-{desi_id}.fits'

    if not os.path.isfile(file):
        #print(f'downloading {desi_id}...')
        print("downloading sv1 brights...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv1/bright/{short_id}/{desi_id}/')
        print("downloading sv1 darks...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv1/dark/{short_id}/{desi_id}/')
        print("downloading sv3 brights...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv3/bright/{short_id}/{desi_id}/')
        print("downloading sv3 darks...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv3/dark/{short_id}/{desi_id}/')
        print("downloading sv3 brights...")
    else:
        print("data is already local.")

    print("done!")

def E_z(z):
    Om = 0.3
    Ol = 0.7
    return np.sqrt( Om * ( 1 + z ) ** 3 + Ol ) # omega_k is 0

def D_c(zf):
    h = .7
    stepSize = 0.001
    steps = int(zf/stepSize)
    hInv = np.zeros(steps)
    zRange = np.linspace(0, zf, num=steps)#np.arange(0, zf, stepSize)

    D_h = 3000/h

    for i in range(len(zRange)):
        hInv[i] = 1./E_z(zRange[i])

    #print(len(hInv),len(zRange))
    out = D_h*np.trapz(hInv, x=zRange)

    return out

def D_m(z):
    return D_c(z)

def D_l_man(z):
    return (1 + z) * D_m(z)


def get_lum(f, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    D_l = cosmo.luminosity_distance(z).cgs.value #this puts dL in cm
    f = f * 1E-17 #flux in erg/cm^2 now
    return f*4*np.pi*D_l**2


def plot_hist_as_line(ax, data, bins=50, color='k', linestyle='-',
                      orientation='vertical', alpha=1.0, linewidth=1.5):
    """
    Plot a histogram as a simple line (straight segments between bin centers).
    No smoothing or interpolation.
    """
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if orientation == "vertical":
        ax.plot(centers, counts, color=color, linestyle=linestyle,
                alpha=alpha, linewidth=linewidth)
    else:  # horizontal
        ax.plot(counts, centers, color=color, linestyle=linestyle,
                alpha=alpha, linewidth=linewidth)


def generate_combined_mask(*masks):
    """
    Creates a new boolean array by combining every array in the masks list using 'and' logic

    :param masks: list: a list with at least one element. Each element is a boolean array of equal length
    :return: A single boolean array that is the 'and' logical combination of all the input arrays
    """
    # masks is a list with at least one element. Each element is a boolean array of equal length
    length = len(masks[0])
    full_mask = np.ones(length, dtype=bool)
    for mask in masks:
        full_mask = np.logical_and(full_mask, mask)
    return full_mask


def read_cigale_results(folder='_full_sky'):
    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'

    #results_1 = pd.read_table(f"{cigale_dir}/9010/out/results.txt", header=0, sep='\s+')
    #results_2 = pd.read_table(f"{cigale_dir}/9011/out/results.txt", header=0, sep='\s+')

    #cigale_results = pd.concat([results_1, results_2], ignore_index=True, sort=False)

    cigale_results = pd.read_table(f"{cigale_dir}/{folder}/out/results.txt", header=0, sep='\s+')

    return cigale_results


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

def extinction_correct(flux, EBV, wavelength):
    # Convert everything to numpy arrays (scalars become 0-d arrays)
    flux = np.asarray(flux, dtype=float)
    EBV = np.asarray(EBV, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)

    # Enforce EBV >= 0
    EBV = np.nan_to_num(EBV, nan=0.0)
    EBV = np.clip(EBV, 0.0, None)

    # Compute extinction
    A = k_lambda_2000(wavelength) * EBV

    return flux * np.exp(0.4 * A)




def testfastqa():
    os.system(f"fastqa --targetids ")