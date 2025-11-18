import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def generate_interpolation_grid():
    # read
    t = ascii.read("apjab16edt2_mrt.txt", format="cds")

    # axes
    logne_vals = np.unique(t['log(ne)'])
    logOH_vals = np.unique(t['logO/H+12'])

    # choose the ratio column you want, e.g. 'OII' (make sure name matches)
    ratio_col = 'OII'   # replace with actual column name in your table

    # build grid: shape must be (len(logne_vals), len(logOH_vals))
    ratio_grid = np.full((len(logne_vals), len(logOH_vals)), np.nan)
    for i, ne in enumerate(logne_vals):
        for j, oh in enumerate(logOH_vals):
            mask = (t['log(ne)'] == ne) & (t['logO/H+12'] == oh)
            if np.any(mask):
                ratio_grid[i, j] = t[mask][ratio_col][0]

    # make sure no NaNs remain
    if np.isnan(ratio_grid).any():
        raise RuntimeError("Grid has missing values -> not purely regular.")

    interp_ratio = RegularGridInterpolator((logne_vals, logOH_vals), ratio_grid,
                                          bounds_error=False, fill_value=None)

    return logne_vals, logOH_vals, ratio_grid, interp_ratio

# inversion function: find logne for obs_ratio and obs_logOH
def find_logne_for_ratio(obs_ratio, obs_logOH, ne_min=None, ne_max=None):

    if ne_min is None: ne_min = logne_vals.min()
    if ne_max is None: ne_max = logne_vals.max()
    f = lambda ne: float(interp_ratio((ne, obs_logOH)) - obs_ratio)
    # Optional: check sign change on bracket
    fa, fb = f(ne_min), f(ne_max)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Requested metallicity outside grid range or extrapolating.")
    if fa == 0:
        return ne_min
    if fb == 0:
        return ne_max
    if fa*fb > 0:
        # no sign change: maybe monotonicity fails or obs_ratio outside grid range
        # return NaN or best-fit using minimization:
        res = root_scalar(lambda x: f(x), x0=(ne_min+ne_max)/2, method='secant')
        if res.converged:
            return res.root
        else:
            return np.nan
    sol = root_scalar(f, bracket=[ne_min, ne_max], method='brentq')
    return sol.root


def plot_grids():

    # --- Plot 1: ratio vs log(ne), one curve per metallicity
    plt.figure(figsize=(7,5))

    for j, oh in enumerate(logOH_vals):
        # Extract the ratio as a function of logne for this metallicity
        ratios = ratio_grid[:, j]
        plt.plot(logne_vals, ratios, label=f'{oh:.2f}')

    a = 0.3771
    b = 2468
    c = 638.4
    R = np.linspace(0.3839, 1.4558, 100)
    ne = np.log10((c * R - a * b) / (a - R))
    plt.plot(ne, R, label='Sanders+16', color='k')

    plt.ylabel('[O II] 3729/3726 line ratio')
    plt.xlabel(r'$log(n_e/cm^{3})$')
    plt.title('Density-sensitive [O II] ratio vs. log($n_e$)')
    plt.legend(title='log(O/H) + 12', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Make fine grids
    ratio_range = np.linspace(np.nanmin(ratio_grid), np.nanmax(ratio_grid), 200)
    logOH_range = np.linspace(logOH_vals.min(), logOH_vals.max(), 200)

    # Allocate array for logne at each (ratio, metallicity)
    # shape: (len(ratio_range), len(logOH_range)) so Y=ratio, X=metallicity
    logne_lookup = np.full((len(ratio_range), len(logOH_range)), np.nan)

    # Invert ratio(logne, logOH) → logne(ratio, logOH)
    for j, oh in enumerate(logOH_range):
        for i, r in enumerate(ratio_range):
            f = lambda ne: interp_ratio((ne, oh)) - r
            try:
                logne_lookup[i, j] = brentq(f, logne_vals.min(), logne_vals.max())
            except ValueError:
                continue  # ratio outside range for this metallicity

    # --- Plot 2 (swapped axes): metallicity on x, ratio on y
    plt.figure(figsize=(7,5))
    X, Y = np.meshgrid(logOH_range, ratio_range)

    # Contours of constant logne
    cs = plt.contour(X, Y, logne_lookup,
                     levels=np.arange(logne_vals.min(), logne_vals.max()+0.5, 0.5),
                     colors='white', linewidths=1)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Color background
    im = plt.imshow(logne_lookup, extent=[logOH_range.min(), logOH_range.max(), ratio_range.min(), ratio_range.max()],
                    origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar(im, label='$log(n_e/cm^3)$')

    plt.xlabel('log(O/H) + 12')
    plt.ylabel('[O II] 3729/3726 line ratio')
    plt.title('Contours of constant electron density')
    plt.tight_layout()
    plt.show()

logne_vals, logOH_vals, ratio_grid, interp_ratio = generate_interpolation_grid()

if __name__ == '__main__':
    plot_grids()