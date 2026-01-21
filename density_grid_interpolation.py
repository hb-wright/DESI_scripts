import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq


def linearly_interpolate_u(x, y, p):
    """
    Linearly interpolates u at point p
    :param x: OIII/OII line ratio
    :param y: log(O/H) + 12 (from R23)
    :param p: pressure
    :return: the ionization parameter log(U) at p, assuming linear U(P)
    """

    A_6 = 13.768
    B_6 = 9.4940
    C_6 = -4.3223
    D_6 = -2.3531
    E_6 = -0.5769
    F_6 = 0.2794
    G_6 = 0.1574
    H_6 = 0.0890
    I_6 = 0.0311
    J_6 = 0.0000

    z_6 = A_6 + B_6*x + C_6*y + D_6*x*y + E_6*(x**2) + F_6*(y**2) + G_6*x*(y**2) + H_6*y*(x**2) + I_6*(x**3) + J_6*(y**3)

    A_7 = -48.953
    B_7 = 6.076
    C_7 = 18.139
    D_7 = -1.4759
    E_7 = -0.4753
    F_7 = -2.3925
    G_7 = 0.1010
    H_7 = 0.0758
    I_7 = 0.0332
    J_7 = 0.1055

    z_7 = A_7 + B_7*x + C_7*y + D_7*x*y + E_7*(x**2) + F_7*(y**2) + G_7*x*(y**2) + H_7*y*(x**2) + I_7*(x**3) + J_7*(y**3)

    return z_6 + (z_7 - z_6) * (p - 6)


def read_pressure_table(filename):
    """
    Read MAPPINGS v5.1 pressure table (MR format).

    Returns:
        logP   : array
        logq   : array
        logOH  : array
        oii    : array
    """

    logP  = []
    logq  = []
    logOH = []
    oii   = []

    with open(filename, "r") as f:
        for line in f:
            # Skip header / separator lines
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            if line.startswith("----"):
                continue
            if not line[0].isdigit():
                continue

            try:
                # Fixed-width slices from byte description
                P   = float(line[0:3])     # bytes 1–3
                q   = float(line[4:8])     # bytes 5–8
                OH  = float(line[9:13])    # bytes 10–13
                OII = float(line[58:65])   # bytes 59–65

                logP.append(P)
                logq.append(q)
                logOH.append(OH)
                oii.append(OII)

            except ValueError:
                # Should not happen for valid rows, but skip defensively
                continue

    return (
        np.array(logP),
        np.array(logq),
        np.array(logOH),
        np.array(oii),
    )

def read_density_table(filename):
    """
    Read MAPPINGS v5.1 pressure table (MR format).

    Returns:
        logne   : array
        logq   : array
        logOH  : array
        oii    : array
    """

    logne  = []
    logq  = []
    logOH = []
    oii   = []

    with open(filename, "r") as f:
        for line in f:
            # Skip header / separator lines
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            if line.startswith("----"):
                continue
            if not line[0].isdigit():
                continue

            try:
                # Fixed-width slices from byte description
                ne  = float(line[0:4])     # bytes 1–4
                q   = float(line[5:9])     # bytes 5–8
                OH  = float(line[10:14])    # bytes 10–13
                OII = float(line[55:62])   # bytes 59–65

                logne.append(ne)
                logq.append(q)
                logOH.append(OH)
                oii.append(OII)

            except ValueError:
                # Should not happen for valid rows, but skip defensively
                continue

    return (
        np.array(logne),
        np.array(logq),
        np.array(logOH),
        np.array(oii),
    )


def build_oii_pressure_interpolator(logP, logq, logOH, oii):
    """
    Builds a RegularGridInterpolator:
        (logP, logq, logOH) -> OII ratio
    """

    P_vals  = np.unique(logP)
    q_vals  = np.unique(logq)
    OH_vals = np.unique(logOH)

    # Create 3D grid
    OII_grid = np.full(
        (len(P_vals), len(q_vals), len(OH_vals)),
        np.nan
    )

    # Fill grid
    for P, q, OH, r in zip(logP, logq, logOH, oii):
        i = np.where(P_vals == P)[0][0]
        j = np.where(q_vals == q)[0][0]
        k = np.where(OH_vals == OH)[0][0]
        OII_grid[i, j, k] = r

    interp = RegularGridInterpolator(
        (P_vals, q_vals, OH_vals),
        OII_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    return interp, P_vals


def build_oii_density_interpolator(logne, logq, logOH, oii):
    """
    Builds a RegularGridInterpolator:
        (logne, logq, logOH) -> OII ratio
    """

    ne_vals = np.unique(logne)
    q_vals  = np.unique(logq)
    OH_vals = np.unique(logOH)

    # Create 3D grid
    OII_grid = np.full(
        (len(ne_vals), len(q_vals), len(OH_vals)),
        np.nan
    )

    # Fill grid
    for P, q, OH, r in zip(logne, logq, logOH, oii):
        i = np.where(ne_vals == P)[0][0]
        j = np.where(q_vals == q)[0][0]
        k = np.where(OH_vals == OH)[0][0]
        OII_grid[i, j, k] = r

    interp = RegularGridInterpolator(
        (ne_vals, q_vals, OH_vals),
        OII_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    return interp, ne_vals


def infer_logP(logq, logOH, oii_obs, oii_interp, P_bounds):
    """
    Infer log(P/k) given log(q), log(O/H)+12, and observed OII ratio.
    """

    def f(logP):
        model = oii_interp((logP, logq, logOH))
        return model - oii_obs

    Pmin, Pmax = P_bounds

    # Check that solution is bracketed
    fmin = f(Pmin)
    fmax = f(Pmax)

    if np.isnan(fmin) or np.isnan(fmax) or fmin * fmax > 0:
        return np.nan

    return brentq(f, Pmin, Pmax)


def infer_logne(logq, logOH, oii_obs, oii_interp, ne_bounds):
    """
    Infer log(P/k) given log(q), log(O/H)+12, and observed OII ratio.
    """

    def f(logne):
        model = oii_interp((logne, logq, logOH))
        return model - oii_obs

    nemin, nemax = ne_bounds

    # Check that solution is bracketed
    fmin = f(nemin)
    fmax = f(nemax)

    if np.isnan(fmin) or np.isnan(fmax) or fmin * fmax > 0:
        return np.nan

    return brentq(f, nemin, nemax)


def example_pressure_interp():
    # Example usage

    # --- one-time setup ---
    logP, logq, logOH, oii = read_pressure_table(
        "apjab16edt1_mrt.txt"
    )

    #print(logP, logq, logOH, oii)

    oii_interp, P_vals = build_oii_pressure_interpolator(
        logP, logq, logOH, oii
    )

    P_bounds = (P_vals.min(), P_vals.max())

    # --- repeated usage ---
    logP_inferred = infer_logP(
        logq=8.25,
        logOH=8.93,
        oii_obs=1.29,
        oii_interp=oii_interp,
        P_bounds=P_bounds
    )
    print(logP_inferred)


def example_density_interp():
    # Example usage

    # --- one-time setup ---
    logne, logq, logOH, oii = read_density_table(
        "apjab16edt2_mrt.txt"
    )

    #print(logP, logq, logOH, oii)

    oii_interp, ne_vals = build_oii_density_interpolator(
        logne, logq, logOH, oii
    )

    ne_bounds = (ne_vals.min(), ne_vals.max())

    # --- repeated usage ---
    logne_inferred = infer_logne(
        logq=8.25,
        logOH=8.93,
        oii_obs=1.29,
        oii_interp=oii_interp,
        ne_bounds=ne_bounds
    )
    print(logne_inferred)


def converge_pressure(metallicity, Roiii, Roii, oii_interp, P_bounds):

    P = 6.5
    delta_P = 10

    while delta_P > 0.1:
        P0 = P
        u = linearly_interpolate_u(Roiii, metallicity, P0)
        q = u + np.log10(3e10)
        # --- repeated usage ---
        P = infer_logP(
            logq=q,
            logOH=metallicity,
            oii_obs=Roii,
            oii_interp=oii_interp,
            P_bounds=P_bounds
        )

        delta_P = P - P0

        print(P, P0, delta_P)

    u = linearly_interpolate_u(Roii, metallicity, P)
    q = u + np.log10(3e10)

    return P, q




def loop_pressure_convergence():
    # --- one-time setup ---
    logP, logq, logOH, oii = read_pressure_table(
        "apjab16edt1_mrt.txt"
    )

    #print(logP, logq, logOH, oii)

    oii_interp, P_vals = build_oii_pressure_interpolator(
        logP, logq, logOH, oii
    )

    P_bounds = (P_vals.min(), P_vals.max())


    eg_metallicity = 8.254858
    eg_5007 = 0.8652226
    eg_3726 = 4.381815
    eg_3729 = 6.229939
    eg_Roiii = eg_5007 / (eg_3726 - eg_3729)
    eg_Roii = 1/0.7040006


    P, q = converge_pressure(eg_metallicity, eg_Roiii, eg_Roii, oii_interp, P_bounds)

    # --- one-time setup ---
    logne, logq, logOH, oii = read_density_table(
        "apjab16edt2_mrt.txt"
    )

    #print(logP, logq, logOH, oii)

    oii_interp, ne_vals = build_oii_density_interpolator(
        logne, logq, logOH, oii
    )

    ne_bounds = (ne_vals.min(), ne_vals.max())

    # --- repeated usage ---
    logne_inferred = infer_logne(
        logq=q,
        logOH=eg_metallicity,
        oii_obs=eg_Roii,
        oii_interp=oii_interp,
        ne_bounds=ne_bounds
    )

    print(logne_inferred, np.log10(22.873705))








if __name__ == "__main__":
    #example_pressure_interp()
    #example_density_interp()
    loop_pressure_convergence()




#############################################################################################
########################### OLD WORK, BEFORE ADDING PRESSURE AXIS ###########################
#############################################################################################




def ionization_param(oii, oiii):
    # Morisset+16
    # We need an equation for ionization parameter that is dependent on pressure

    log_u = -2.74 - (1.0 * np.log10(oii/oiii))


def pressure_interpolation_grid():
    # read
    t = ascii.read("apjab16edt1_mrt.txt", format="cds")

    # axes
    logne_vals = np.unique(t['log(ne)'])
    logOH_vals = np.unique(t['logO/H+12'])
    logU_vals = np.unique(t['log(q)'])

    # choose the ratio column you want, e.g. 'OII' (make sure name matches)
    ratio_col = 'OII'   # replace with actual column name in your table

    # build grid: shape must be (len(logne_vals), len(logOH_vals))
    ratio_grid = np.full(
        (len(logne_vals), len(logOH_vals), len(logU_vals)),
        np.nan
    )
    for i, ne in enumerate(logne_vals):
        for j, oh in enumerate(logOH_vals):
            for k, logU in enumerate(logU_vals):
                mask = (
                        (t['log(ne)'] == ne) &
                        (t['logO/H+12'] == oh) &
                        (t['log(q)'] == logU)
                )
                if np.any(mask):
                    ratio_grid[i, j, k] = t[mask][ratio_col][0]

    # make sure no NaNs remain
    if np.isnan(ratio_grid).any():
        raise RuntimeError("Grid has missing values -> not purely regular.")

    interp_ratio = RegularGridInterpolator(
        (logne_vals, logOH_vals, logU_vals),
        ratio_grid,
        bounds_error=False,
        fill_value=None
    )

    return logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio





def density_interpolation_grid():
    # read
    t = ascii.read("apjab16edt2_mrt.txt", format="cds")

    # axes
    logne_vals = np.unique(t['log(ne)'])
    logOH_vals = np.unique(t['logO/H+12'])
    logU_vals = np.unique(t['log(q)'])

    # choose the ratio column you want, e.g. 'OII' (make sure name matches)
    ratio_col = 'OII'   # replace with actual column name in your table

    # build grid: shape must be (len(logne_vals), len(logOH_vals))
    ratio_grid = np.full(
        (len(logne_vals), len(logOH_vals), len(logU_vals)),
        np.nan
    )
    for i, ne in enumerate(logne_vals):
        for j, oh in enumerate(logOH_vals):
            for k, logU in enumerate(logU_vals):
                mask = (
                        (t['log(ne)'] == ne) &
                        (t['logO/H+12'] == oh) &
                        (t['log(q)'] == logU)
                )
                if np.any(mask):
                    ratio_grid[i, j, k] = t[mask][ratio_col][0]

    # make sure no NaNs remain
    if np.isnan(ratio_grid).any():
        raise RuntimeError("Grid has missing values -> not purely regular.")

    interp_ratio = RegularGridInterpolator(
        (logne_vals, logOH_vals, logU_vals),
        ratio_grid,
        bounds_error=False,
        fill_value=None
    )

    return logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio


# inversion function: find logne for obs_ratio and obs_logOH
def find_logne_for_ratio(obs_ratio, obs_logOH, logU, ne_min=None, ne_max=None):

    if ne_min is None: ne_min = logne_vals.min()
    if ne_max is None: ne_max = logne_vals.max()
    f = lambda ne: float(
        interp_ratio((ne, obs_logOH, logU)) - obs_ratio
    )
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

    sol = root_scalar(
        f,
        bracket=[logne_vals.min(), logne_vals.max()],
        method='brentq'
    )

    return sol.root


def plot_grids(logU_plot=6.5):

    kU = np.argmin(np.abs(logU_vals - logU_plot))

    # --- Plot 1: ratio vs log(ne), one curve per metallicity
    plt.figure(figsize=(7,5))

    for j, oh in enumerate(logOH_vals):
        # Extract the ratio as a function of logne for this metallicity
        ratios = ratio_grid[:, j, kU]
        plt.plot(logne_vals, ratios, label=f'{oh:.2f}')

    a = 0.3771
    b = 2468
    c = 638.4
    R = np.linspace(0.3839, 1.4558, 100)
    ne = np.log10((c * R - a * b) / (a - R))
    plt.plot(ne, R, label='Sanders+16', color='k')

    plt.ylabel('[O II] 3729/3726 line ratio')
    plt.xlabel(r'$log(n_e/cm^{3})$')
    plt.title(f'Density-sensitive [O II] ratio (log U = {logU_plot:.2f})')
    plt.legend(title='log(O/H) + 12', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'paper_figures/density_grid_{logU_plot:.2f}.png')
    plt.show()

    # --- Make fine grids
    ratio_range = np.linspace(np.nanmin(ratio_grid), np.nanmax(ratio_grid), 200)
    logOH_range = np.linspace(logOH_vals.min(), logOH_vals.max(), 200)
    logU_range = np.linspace(logU_vals.min(), logU_vals.max(), 200)

    # Allocate array for logne at each (ratio, metallicity)
    # shape: (len(ratio_range), len(logOH_range)) so Y=ratio, X=metallicity
    logne_lookup = np.full((len(ratio_range), len(logOH_range)), np.nan)

    # Invert ratio(logne, logOH) → logne(ratio, logOH)
    for j, oh in enumerate(logOH_range):
        for i, r in enumerate(ratio_range):
            f = lambda ne: interp_ratio((ne, oh, logU_plot)) - r
            try:
                logne_lookup[i, j] = brentq(f, logne_vals.min(), logne_vals.max())
            except ValueError:
                continue  # ratio outside range for this metallicity

    # --- Plot 2: metallicity on x, ratio on y
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
    plt.title(f'Contours of constant electron density (log U = {logU_plot:.2f})')
    plt.tight_layout()
    plt.savefig(f'paper_figures/density_grid_interpolation_{logU_plot:.2f}.png', dpi=300)
    plt.show()


def plot_all_curves():

    linestyles = [(0, (5, 10)), '--', (0, (5, 1)), (0, (1, 10)), ':', (0, (1, 1)), (0, (3, 10, 1, 10)), '-.', (0, (3, 1, 1, 1))]
    colors = ['b', 'g', 'r', 'c', 'm']

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- main grid
    for i, logU_plot in enumerate(np.arange(6.5, 8.75, 0.25)):
        kU = np.argmin(np.abs(logU_vals - logU_plot))

        for j, oh in enumerate(logOH_vals):
            # Extract the ratio as a function of logne for this metallicity
            ratios = ratio_grid[:, j, kU]
            plt.plot(logne_vals, ratios, label=f'{oh:.2f}, {logU_plot:.2f}', linestyle=linestyles[i], color=colors[j], linewidth=1)

    a = 0.3771
    b = 2468
    c = 638.4
    R = np.linspace(0.3839, 1.4558, 100)
    ne = np.log10((c * R - a * b) / (a - R))
    plt.plot(ne, R, label='Sanders+16', color='k')

    plt.ylabel('[O II] 3729/3726 line ratio', fontsize=20)
    plt.xlabel(r'$log(n_e/cm^{3})$', fontsize=20)
    plt.title(f'Density-sensitive [O II] ratio', fontsize=20)
    # --- proxy handles for log(U) (linestyle only)
    style_handles = [
        Line2D([0], [0], color='k', lw=2, linestyle=ls)
        for ls in linestyles
    ]
    style_labels = [f'{u:.2f}' for u in logU_vals[:len(linestyles)]]

    leg1 = ax.legend(
        style_handles,
        style_labels,
        title='log(U)',
        bbox_to_anchor=(1.03, .7),
        loc='upper left',
        fontsize=14,
    )
    ax.add_artist(leg1)  # keep this legend when adding the next

    # --- proxy handles for metallicity (color only)
    color_handles = [
        Line2D([0], [0], color=c, lw=2)
        for c in colors
    ]
    color_labels = [f'{oh:.2f}' for oh in logOH_vals[:len(colors)]]

    leg2 = ax.legend(
        color_handles,
        color_labels,
        title='12 + log(O/H)',
        bbox_to_anchor=(1.03, .98),  # vertical offset → whitespace
        loc='upper left',
        fontsize=14,
    )

    #plt.legend(title='log(O/H) + 12, log(U)', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=0.83)
    plt.savefig(f'paper_figures/all_density_grid.png')
    plt.show()
    return 0

#logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio = density_interpolation_grid()


#if __name__ == '__main__':
    #for logU in np.arange(6.5, 8.75, 0.25):
    #    plot_grids(logU_plot=logU)
    #plot_all_curves()