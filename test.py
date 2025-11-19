# Fit the first derivative of a melting curve to a 2-state model
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from scipy import optimize
import matplotlib.pyplot as plt   # <--- NEW for plotting

# ---------------- User settings ----------------
conc_ds               = 2.5e-6  # strand concentration (M) for bimolecular models; ignored if intramolecular
is_truncate           = False    # truncate only for qPCR data
temp_truncate         = 30      # Truncate the data at Tm - temp_truncate (°C)
fraction_init         = 0.10    # fraction of points at each end for baseline initial guesses
nm_options            = dict(maxiter=1e6,
                             maxfev=1e6,
                             xatol=1e-6,
                             fatol=1e-6
                             )   # Nelder-Mead minimization options
# ------------------------------------------------

R = 8.314  # ideal gas constant in J/(mol·K)

@dataclass
class MeltParams:
    delta_h: float   # change in enthalpy (J/mol)
    delta_s: float   # change in entropy (J/mol/K)
    m_ds: float      # slope of the linear baseline of the folded state
    m_ss: float      # slope of the linear baseline of the unfolded state
    b_ds: float      # Y-intercept of the linear baseline of the folded state
    b_ss: float      # Y-intercept of the linear baseline of the unfolded state

# Initial parameters (kept for reference, not used directly)
params_init = MeltParams(
    delta_h     = -530.6e3,  # -600 kJ/mol
    delta_s     = -1421,     # J/mol/K
    m_ds        = -409.6852903,
    m_ss        = -22.2040091,
    b_ds        = 158426.7464,
    b_ss        = 10801.31894,
)

def params_from_array(x: np.ndarray) -> MeltParams:
    """
    Map a flat parameter vector to MeltParams.

    We *define* the order once here:
        x = [delta_h, delta_s, m_ds, m_ss, b_ds, b_ss]
    and reuse it everywhere to avoid confusion.
    """
    delta_h, delta_s, m_ds, m_ss, b_ds, b_ss = x
    return MeltParams(delta_h, delta_s, m_ds, m_ss, b_ds, b_ss)

# Functions --------------------------------------
def calc_ka(temp: np.ndarray, params: MeltParams) -> np.ndarray:
    """
    Calculate equilibrium constant Ka for A + B -> C, defined as Ka = [C]/[A][B].
    :param temp: temperature in Kelvin
    :param params: delta_h, delta_s
    :return: equilibrium constant in /mol
    """
    return np.exp(params.delta_s / R - params.delta_h / (R * temp))


def calc_frac_unfold(temp: np.ndarray, params: MeltParams, conc: float) -> np.ndarray:
    """
    Calculate the fraction of unfolded duplex.
    :param conc: concentration of the duplex
    :param temp: temperature in Kelvin
    :param params: delta_h, delta_s, conc
    :return: fraction of unfolded duplex (0 - 1)
    """
    k_a = calc_ka(temp, params)
    return (np.sqrt(1 + 4 * k_a * conc) - 1) / (2 * k_a * conc)


def calc_rsp(temp: np.ndarray, params: MeltParams, conc: float) -> np.ndarray:
    """
    Take the `temperature` and `params` to simulate the response of the
    machine (RFU or Abs) of duplex melting
    :param conc: concentration of the duplex
    :param temp: temperature in Kelvin
    :param params: d delta_h, delta_s, conc, m_ds, m_ss, b_ds, b_ss
    :return: response of the machine (RFU or Abs)
    """
    frac_unfold = calc_frac_unfold(temp, params, conc)
    return (params.b_ds + params.m_ds * temp + frac_unfold
            * (params.b_ss - params.b_ds + (params.m_ss - params.m_ds) * temp)
            )

def calc_rsp_deriv(temp: np.ndarray, params: MeltParams, conc: float) -> np.ndarray:
    """
       Take the `temperature` and `params` to simulate the first derivative
       of response of the machine (RFU or Abs) of duplex melting
       :param conc: concentration of the duplex
       :param temp: temperature in Kelvin
       :param params: delta_h, delta_s, conc, m_ds, m_ss, b_ds, b_ss
       :return: derivative of the response with respect to T
       """
    k_a = calc_ka(temp, params)
    frac_unfold = calc_frac_unfold(temp, params, conc)
    return (params.m_ds + (params.m_ss - params.m_ds) * frac_unfold
            + ((params.b_ss - params.b_ds) + (params.m_ss - params.m_ds) * temp)
            * (params.delta_h / (R * temp ** 2))
            * (frac_unfold - 1) / np.sqrt(1 + 4 * k_a * conc ))

def calc_t_m_anal(params: MeltParams, conc: float) -> float:
    """
    Calculate analytic Tm (°C) based on DeltaH, DeltaS and conc.
    :param conc: concentration of the duplex
    :param params: MeltParams with delta_h, delta_s
    :return: thermal melting temperature in Celsius
    """
    ln_k_a = np.log(2.0 / conc)

    denom = params.delta_s - R * ln_k_a
    if not np.isfinite(denom) or abs(denom) < 1e-16:
        return np.nan
    t_m = params.delta_h / denom
    return float(t_m - 273.15)


def calc_merit_rsp(x: np.ndarray, temp: np.ndarray, conc: float, rsp_exp: np.ndarray) -> float:
    """
    RSS merit function on response.
    x is the parameter vector [delta_h, delta_s, m_ds, m_ss, b_ds, b_ss]
    """
    params = params_from_array(x)
    rsp_fit = calc_rsp(temp, params, conc)
    resid = rsp_fit - rsp_exp
    return float(np.nansum(resid**2))  # <-- scalar float


def calc_merit_deriv(x: np.ndarray, temp: np.ndarray, conc: float, rsp_deriv_exp: np.ndarray) -> float:
    """
    RSS merit function on first derivative.
    """
    params = params_from_array(x)
    rsp_deriv_fit = calc_rsp_deriv(temp, params, conc)
    resid = rsp_deriv_fit - rsp_deriv_exp
    return float(np.nansum(resid**2))


def plot_overlay_and_residuals(temp_c: np.ndarray,
                               y_exp: np.ndarray,
                               y_fit: np.ndarray,
                               residual: np.ndarray,
                               title: str,
                               ylabel: str,
                               out_path: str):
    """
    Utility to make a 2-panel figure:
      top:   experimental vs fitted
      bottom: residuals (fit - exp)
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax = axes[0]
    ax.plot(temp_c, y_exp, label="Experimental")
    ax.plot(temp_c, y_fit, label="Fit")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    axr = axes[1]
    axr.plot(temp_c, residual, label="Residual")
    axr.axhline(0.0, color="k", lw=1, ls="--")
    axr.set_xlabel("Temperature (°C)")
    axr.set_ylabel("Fit - Exp")
    axr.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Open a CSV file---------------------------------
# root = Tk()
# root.withdraw()  # Hide the root window
# file_path = filedialog.askopenfilename(
#     filetypes=[("CSV files", "*.csv")],
#     title="Select a file containing melting curves"
# )
# root.destroy()
#
# if not file_path:
#     print("No file selected!")
#     exit()

file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_EG_2.5uM.csv"
# file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_UV_2.5uM.csv"
# Extract filename
dir_name = os.path.dirname(file_path)
file_base = os.path.splitext(os.path.basename(file_path))[0]
print(f"File name: {file_base}")

# Read CSV into a DataFrame ----------------------
tbl = pd.read_csv(file_path)

# Get column names
columns = tbl.columns.tolist()

# Find the index of the column 'Temperature'
if "Temperature" not in columns:
    print('The file must contain a header named "Temperature"!')
    exit()
else:
    idx_temp = columns.index("Temperature")

# Extract the temperature column by idx_temp and convert it to numeric values
temp_celsius = pd.to_numeric(tbl.iloc[:, idx_temp], errors="coerce").to_numpy()
temp_kelvin = temp_celsius + 273.15  # Convert the temperature to Kelvin

data_columns = [i for i in range(idx_temp, tbl.shape[1]) if i != idx_temp]
well_ID = columns[idx_temp + 1:]  # well ID names or sample names

# Preallocate data matrix
rsp_all = np.full((len(temp_kelvin), len(data_columns)), np.nan, dtype=float)

# Build the data matrix
for index, value in enumerate(data_columns):
    col = pd.to_numeric(tbl.iloc[:, value], errors='coerce').to_numpy()
    rsp_all[:, index] = col

n_curves = rsp_all.shape[1]
n_temp = len(temp_kelvin)

# Prepare output files ---------------------------
curves_out = {"Temperature": temp_celsius.copy()}
params_rows = []

fig_dir = os.path.join(dir_name, "figures")

# Fit each curve ---------------------------------
for i in range(n_curves):

    # Experimental data for this column
    rsp_indiv = rsp_all[:, i]

    # Tm from derivative (on full curve)
    rsp_deriv = np.gradient(rsp_indiv, temp_kelvin)
    t_m_deriv_kelvin = np.round(
                float(temp_kelvin[np.argmax(np.abs(rsp_deriv))]),
                decimals=1
                )

    # Determine the range of the lower temperature baseline
    # Lower temperature baseline of qPCR drifts significantly
    # Only fit the data in the temperature range of
    # [Tm - temp_truncate:] (arbitrarily determined from the experimental data)
    # BUGFIX: Tm is already in Kelvin, so subtract temp_truncate directly
    temp_start = t_m_deriv_kelvin - temp_truncate

    if is_truncate and temp_start >= 288.15:  # 15 C in Kelvin
        idx_start = np.abs(temp_kelvin - temp_start).argmin()
    else:
        idx_start = 0

    # Slice the array to truncate the data
    temp_sliced = temp_kelvin[idx_start:]
    rsp_sliced = rsp_indiv[idx_start:]
    rsp_deriv_sliced = rsp_deriv[idx_start:]

    # Initial baselines from ends (low T ~ ds; high T ~ ss)
    n = len(temp_sliced)

    # take the first and last 10% of data points for initial baselines
    k_end = int(round(fraction_init * n))
    m_ds_init, b_ds_init = np.polyfit(temp_sliced[:k_end], rsp_sliced[:k_end], deg=1)
    m_ss_init, b_ss_init = np.polyfit(temp_sliced[n - k_end:], rsp_sliced[n - k_end:], deg=1)

    # Initial thermodynamic parameters (association: ΔH<0, ΔS<0)
    dH_init = -200.0e3  # J/mol
    dS_init = -600.0  # J/mol/K

    # Fitting directly to the response curve----------------------------
    # Initial parameters as a vector [dH, dS, m_ds, m_ss, b_ds, b_ss]
    x0_rsp = np.asarray([dH_init, dS_init, m_ds_init, m_ss_init, b_ds_init, b_ss_init ], dtype=float)

    res = optimize.minimize(
        calc_merit_rsp,
        x0_rsp,
        args=(temp_sliced, conc_ds, rsp_sliced),
        method="Nelder-Mead",
        options=nm_options
    )

    # Return the fitted parameters as a MeltParams object
    params_rsp = params_from_array(res.x)
    rsp_fit_sliced = calc_rsp(temp_sliced, params_rsp, conc_ds)
    rss = float(np.nansum((rsp_fit_sliced - rsp_sliced) ** 2))

    # ----- Pad fitted/residual back to full length with NaNs so columns align -----
    rsp_fit_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    residual_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    rsp_fit_full[idx_start:] = rsp_fit_sliced
    residual_full[idx_start:] = (rsp_fit_sliced - rsp_sliced)

    # Tm from fitted dH, dS, and C
    t_m_anal = np.round(calc_t_m_anal(params_rsp, conc_ds), decimals=1)

    # Fitting to the first derivative curve-----------------------------
    x0_deriv = res.x.copy()

    res_deriv = optimize.minimize(
        calc_merit_deriv,
        x0_deriv,
        args=(temp_sliced, conc_ds, rsp_deriv_sliced),
        method="Nelder-Mead",
        options=nm_options
    )

    # Return the fitted parameters (derivative fit)
    params_deriv = params_from_array(res_deriv.x)
    rsp_deriv_fit_sliced = calc_rsp_deriv(temp_sliced, params_deriv, conc_ds)
    rss_deriv = float(np.nansum((rsp_deriv_fit_sliced - rsp_deriv_sliced) ** 2))

    # ----- Pad fitted/residual back to full length with NaNs so columns align -----
    rsp_deriv_fit_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    residual_deriv_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    rsp_deriv_fit_full[idx_start:] = rsp_deriv_fit_sliced
    residual_deriv_full[idx_start:] = (rsp_deriv_fit_sliced - rsp_deriv_sliced)

    # Tm from fitted dH, dS, and C (derivative fit)
    t_m_deriv_anal = np.round(calc_t_m_anal(params_deriv, conc_ds), decimals=1)

    # Assemble wide outputs
    curves_out[f"{well_ID[i]}_exp"] = rsp_indiv
    curves_out[f"{well_ID[i]}_fitted"] = rsp_fit_full
    curves_out[f"{well_ID[i]}_residual"] = residual_full
    curves_out[f"{well_ID[i]}_deriv"] = rsp_deriv
    curves_out[f"{well_ID[i]}_deriv_fitted"] = rsp_deriv_fit_full
    curves_out[f"{well_ID[i]}_deriv_residual"] = residual_deriv_full

    # --- Parameter row
    params_rows.append({
        "curve_id": str(well_ID[i]),
        "DeltaH (kJ/mol)": float(np.round(params_rsp.delta_h / 1000, decimals=1)),
        "DeltaS (J/mol/K)": float(np.round(params_rsp.delta_s, decimals=1)),
        "Tm (C)": float(t_m_anal) if np.isfinite(t_m_anal) else np.nan,
        "RSS": rss,
        "DeltaH_deriv (kJ/mol)": float(np.round(params_deriv.delta_h / 1000, decimals=1)),
        "DeltaS_deriv (J/mol/K)": float(np.round(params_deriv.delta_s, decimals=1)),
        "Tm_deriv_anal (C)": float(t_m_deriv_anal) if np.isfinite(t_m_deriv_anal) else np.nan,
        "Tm_deriv_true (C)": float(t_m_deriv_kelvin - 273.15) if np.isfinite(t_m_deriv_anal) else np.nan,
    })

    # --- Figures for this curve --------------------------------------
    # a) Response fit
    plot_overlay_and_residuals(
        temp_celsius,
        rsp_indiv,
        rsp_fit_full,
        residual_full,
        title=f"{well_ID[i]}: response fit (Tm_anal={t_m_anal:.1f} °C)",
        ylabel="Response",
        out_path=os.path.join(dir_name, "figures", f"{file_base}_{well_ID[i]}_response_fit.png"),
    )

    # b) Derivative fit (experimental derivative vs fit)
    plot_overlay_and_residuals(
        temp_celsius,
        rsp_deriv,
        rsp_deriv_fit_full,
        residual_deriv_full,
        title=f"{well_ID[i]}: derivative fit (Tm_anal={t_m_deriv_anal:.1f} °C)",
        ylabel="d(Response)/dT",
        out_path=os.path.join(dir_name, "figures", f"{file_base}_{well_ID[i]}_derivative_fit.png"),
    )

# ---------- Write outputs ----------
curves_df = pd.DataFrame(curves_out)
params_df = pd.DataFrame(params_rows,
                         columns=["curve_id", "DeltaH (kJ/mol)",
                                  "DeltaS (J/mol/K)", "Tm (C)",
                                  "RSS", "DeltaH_deriv (kJ/mol)",
                                  "DeltaS_deriv (J/mol/K)",
                                  "Tm_deriv_anal (C)", "Tm_deriv_true (C)",
                                  ])

curves_out_path = os.path.join(dir_name, f"{file_base}_fitted_curves.csv")
params_out_path = os.path.join(dir_name, f"{file_base}_fitted_parameters.csv")

curves_df.to_csv(curves_out_path, index=False)
params_df.to_csv(params_out_path, index=False)

print("Done.")
print("Wrote:")
print("  ", curves_out_path)
print("  ", params_out_path)
print("  figures (per-curve response/derivative fits) in:", os.path.join(dir_name, "figures"))