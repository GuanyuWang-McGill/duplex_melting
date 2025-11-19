# Fit melt curves directly on RFU or absorbance with a 2-state model,
# give out DeltaH, DeltaS and Tm
#
# INPUT CSV: first row is headers. One header is "Temperature"
# It should be from lower temperature to higher temperature
# All other numeric columns are treated as sample wells
#
# OUTPUTS (for input 'name.csv'):
#   name_fitted_curves.csv:
#       Temperature, <id>_exp, <id>_fitted, <id>_residual, ...
#   name_fitted_parameters.csv:
#       curve_id, DeltaH_kJ_per_mol, DeltaS_J_per_molK, Tm_C, RSS, Tm_derivative
#
# Model:
#   Ka = exp(ΔS/R - ΔH/(R*T)), T in Kelvin; R = 8.314 J/mol/K
#   x (folded fraction) from mass action (equimolar strands):
#       Intramolecular (hairpin):   Ka*(1 - x) = x
#       Self-complementary:         2*Ka*C*(1 - x)^2 = x
#       Non-self:                   Ka*C*(1 - x)^2 = x
#   C is the initial concentration of the strand
#
#   RFU = x*(b_ds + m_ds*T) + (1 - x)*(b_ss + m_ss*T)
#
# Optimizer: fminsearch (Nelder–Mead), merit = RSS.
#
# Import libraries:
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from scipy import optimize
import matplotlib.pyplot as plt

# Experiment settings --------------------------------------------------
conc_ds               = 2.5e-6   # strand concentration (M) for bimolecular models; ignored if intramolecular
is_intramolecular     = False    # True for hairpins (unimolecular)
is_self_complementary = False    # True for self-complementary duplexes (homodimer)
method                = "UV" # What method (UV-Vis, qPCR) and dye (SG, EG) have been used

# Fitting settings -----------------------------------------------------
is_truncate           = False    # truncate only for qPCR data
temp_truncate         = 30       # Truncate the data at Tm - temp_truncate
fraction_init         = 0.10     # fraction (10%) of points at each end for baseline initial guesses
nm_options            = dict(maxiter=1e6,
                             maxfev=1e6,
                             xatol=1e-6,
                             fatol=1e-6
                             )   # Nelder-Mead minimization options
# ----------------------------------------------------------------------

R = 8.314  # ideal gas constant in J/(mol·K)

# Define thermodynamic parameters---------------------------------------
@dataclass
class ThermoParams:
    delta_h: float   # change in enthalpy (J/mol)
    delta_s: float   # change in entropy (J/mol/K)
    b_ds: float      # Y-intercept of the linear baseline of the folded state
    m_ds: float      # slope of the linear baseline of the folded state
    b_ss: float      # Y-intercept of the linear baseline of the unfolded state
    m_ss: float      # slope of the linear baseline of the unfolded state

# Functions ------------------------------------------------------------
# Biophysical Models ---------------------------------------------------
def calc_ka(temp: np.ndarray, params: ThermoParams) -> np.ndarray:
    """
    Calculate the association constant Ka
        For intramoleular duplex hybridization A <-> C, Ka = [C]/[A]
        For bimolecular self-complementary duplex hybridization A + A <-> C, Ka = [C]/[A]^
        For bimolecular duplex hybridization A + B <-> C, Ka = [C]/[A][B]
    :return: equilibrium constant in mol-1
    """
    return np.exp(params.delta_s / R - params.delta_h / (R * temp))


def calc_frac_fold(temp: np.ndarray,
                   conc: float,
                   params: ThermoParams
                   ) -> np.ndarray:
    """
    Calculate fraction folded x(T):
        - Intramolecular (A ⇌ C):       x = K / (1 + K),  independent of C
        - Non-self (A + B ⇌ C, eqm):    K*C*(1 - x)^2 = x
        - Self-complementary (2A ⇌ C):  2*K*C*(1 - x)^2 = x
    :return: fraction of folded duplex (0 - 1)
    """
    ka = calc_ka(temp, params)

    if is_intramolecular:
        x = ka / (1.0 + ka)
        return np.clip(x, 0.0, 1.0) # Limit the x to between 0 and 1

    ka_conc = np.maximum(ka * conc, np.finfo(float).tiny) # Make sure that Ka * C is never 0

    if not is_self_complementary: # (A + B ⇌ C, eqm)
        # Non-self: u = 1 - x,  K*C*y^2 + u - 1 = 0  ->  u = (-1 + sqrt(1 + 4KC)) / (2KC)
        u = (-1.0 + np.sqrt(1.0 + 4.0 * ka_conc)) / (2.0 * ka_conc)
        x = 1.0 - u
    else: # (2A ⇌ C, eqm)
        # Self: u = 1 - x,  2*K*C*u^2 + u - 1 = 0  ->  u = (-1 + sqrt(1 + 8KC)) / (4KC)
        u = (-1.0 + np.sqrt(1.0 + 8.0 * ka_conc)) / (4.0 * ka_conc)
        x = 1.0 - u
    return np.clip(x, 0.0, 1.0)


def calc_rsp(temp: np.ndarray,
             conc: float,
             params: ThermoParams,
             ) -> np.ndarray:
    """
    Take the `temperature`, `concentration`, and `params` to simulate the response
    of the machine (RFU or Abs).
    params = [ΔH, ΔS, b_ds, m_ds, b_ss, m_ss]
    :return: simulated response
    """
    frac_fold = calc_frac_fold(temp, conc, params)

    return (params.b_ds + params.m_ds * temp + (1- frac_fold)
            * (params.b_ss - params.b_ds + (params.m_ss - params.m_ds) * temp))


def calc_rsp_deriv(temp: np.ndarray,
                   conc: float,
                   params: ThermoParams,
                   ) -> np.ndarray:
    """
    Take the `temperature` and `params` to simulate the first derivative
    of response of the machine (RFU or Abs) of duplex melting
    :return: derivative of the response with respect to T
    """
    frac_fold = calc_frac_fold(temp, conc, params)

    ka = calc_ka(temp, params)

    return (params.m_ds + (params.m_ss - params.m_ds) * (1 - frac_fold)
            - ((params.b_ss - params.b_ds) + (params.m_ss - params.m_ds) * temp)
            * (params.delta_h / (R * temp ** 2))
            * (frac_fold / np.sqrt(1 + 4 * ka * conc )))


def calc_tm_anal(conc: float,
                 params: ThermoParams,
                 ) -> float:
    """
    Calculate analytic Tm (°C) based on stoichiometry.
       Intramolecular: Ka(Tm)=1  -> lnKa=0
       Non-self:       Ka(Tm)=2/C
       Self:           Ka(Tm)=1/C
    :return: thermal melting temperature in Kelvin
    """
    if is_intramolecular:
        ln_ka = 0.0
    elif is_self_complementary:
        ln_ka = np.log(1.0 / conc)
    else:
        ln_ka = np.log(2.0 / conc)

    denom = params.delta_s - R * ln_ka

    if not np.isfinite(denom) or abs(denom) < 1e-16:
        return np.nan

    tm_kelvin = params.delta_h / denom
    return float(tm_kelvin)


def calc_tm_estimate(temp: np.ndarray,
                     rsp_exp: np.ndarray
                     ) -> float:
    """Estimate Tm (K) based on the maximum of the derivative."""
    derivative = np.gradient(rsp_exp, temp)
    return float(temp[np.argmax(np.abs(derivative))])


# Fitting Algorithm ----------------------------------------------------
def merit_rsp(params_fit: ThermoParams,
              temp: np.ndarray,
              conc: float,
              rsp_exp: np.ndarray,
              ) -> float:
    """
    RSS merit function on response.
    parms-fit = [ΔH, ΔS, b_ds, m_ds, b_ss, m_ss]
    """
    params = params_from_array(params_fit)
    rsp_fit = calc_rsp(temp, conc, params)

    resid = rsp_fit - rsp_exp
    return float(np.nansum(resid**2))  # <-- scalar float


def merit_rsp_deriv(params_fit: ThermoParams,
                    temp: np.ndarray,
                    conc: float,
                    rsp_deriv_exp: np.ndarray,
                    ) -> float:
    """
    RSS merit function on first derivative.
    parms-fit is the parameter vector [delta_h, delta_s, m_ds, m_ss, b_ds, b_ss]
    """
    params = params_from_array(params_fit)
    rsp_deriv_fit = calc_rsp_deriv(temp, conc, params)

    resid = rsp_deriv_fit - rsp_deriv_exp
    return float(np.nansum(resid**2))


# Other Helper Functions------------------------------------------------
def params_from_array(param_array: np.ndarray) -> ThermoParams:
    """
    Map a flat parameter vector to ThermoParams.
    The order once here:
        x = [delta_h, delta_s, b_ds, m_ds, b_ss, m_ss]
    and reuse it everywhere to avoid confusion.
    """
    delta_h, delta_s, b_ds, m_ds, b_ss, m_ss = param_array
    return ThermoParams(delta_h, delta_s, b_ds, m_ds, b_ss, m_ss)


def plot_overlay(temp_c: np.ndarray,
                 y_exp: np.ndarray,
                 y_fit: np.ndarray,
                 residual: np.ndarray,
                 title: str,
                 ylabel: str,
                 method: str,
                 data_type: str,
                 out_path: str,
                 ) -> None:
    """
    Make a 2-panel PNG:
      - top: experimental (open circles) vs fitted (line)
      - bottom: residuals (fit - exp)

    bottom_height_ratio: fraction of total height used for the residual panel
                         e.g., 0.3 = bottom is 30%, top is 70%
    """

    # Create a figure
    fig = plt.figure(figsize=(6, 6))

    # Ratio: top gets (1 - bottom_ratio), bottom gets bottom_ratio
    bottom_height_ratio = 0.15
    gs = plt.GridSpec(2, 1,
                      height_ratios=[1 - bottom_height_ratio, bottom_height_ratio]
                      )

    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    # -------- top panel --------
    ax.plot(temp_c, y_exp,
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="red",
            label="Experimental Data"
            )

    ax.plot(temp_c, y_fit,
            linestyle="-", linewidth=1.3,
            color="black",
            label="Fitted Curve"
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(False)

    # No inner grid, black borders
    for spine in ax.spines.values():
        spine.set_color("black")

    # Hide tick labels and axis label
    ax.tick_params(labelbottom=False)  # hide tick labels on top plot
    ax.xaxis.label.set_visible(False)  # hide x-axis label on top plot

    # -------- bottom panel --------
    axr.plot(temp_c, residual,
             linestyle="none",
             marker="o", markersize=4,
             markerfacecolor="none",
             markeredgecolor="red"
             )

    axr.axhline(0, color="black", lw=0.8, ls="--")
    axr.set_xlabel("Temperature (°C)")
    axr.set_ylabel("Residual")
    axr.grid(False)
    for spine in axr.spines.values():
        spine.set_color("black")

    # Set the Y axis of the residual plots
    if data_type == "direct":
        if "uv" in method.casefold():
            axr.set_yticks([-0.004, 0, 0.004])
        elif "qpcr" in method.casefold():
            axr.set_yticks([-1000, 0, 1000])
    elif data_type == "derivative":
        if "uv" in method.casefold():
            axr.set_yticks([-0.001, 0, 0.001])
        elif "qpcr" in method.casefold():
            axr.set_yticks([-500, 0, 500])


    axr.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# File Handling---------------------------------------------------------
# Open a CSV file-------------------------------------------------------
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

#file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_EG_2.5uM.csv"
file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_UV_2.5uM.csv"
# Extract filename------------------------------------------------------
dir_name = os.path.dirname(file_path)
file_base = os.path.splitext(os.path.basename(file_path))[0]
print(f"File name: {file_base}")

# Read CSV into a DataFrame --------------------------------------------
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

# Assign the name of each column as well_ID.
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

# Prepare output files -------------------------------------------------
curves_out = {"Temperature": temp_celsius.copy()}
params_rows = []
fig_dir = os.path.join(dir_name, "figures")

# Fit each curve -------------------------------------------------------
for i in range(n_curves):

    # Experimental data for this column
    rsp_exp = rsp_all[:, i]

    # First derivative curve
    rsp_deriv = np.gradient(rsp_exp, temp_kelvin)

    # Tm estimation from derivative
    tm_estim = np.round(calc_tm_estimate(temp_kelvin, rsp_exp), decimals=1)

    # Determine the range of the lower temperature baseline
    # Lower temperature baseline of qPCR drifts significantly
    # Only fit the data in the temperature range of
    # [Tm - temp_truncate:] (arbitrarily determined from the experimental data)
    temp_start = tm_estim - temp_truncate

    if is_truncate and temp_start >= 288.15:  # 15 C in Kelvin
        idx_start = np.abs(temp_kelvin - temp_start).argmin()
    else:
        idx_start = 0

    # Slice the array to truncate the data
    temp_sliced = temp_kelvin[idx_start:]
    rsp_sliced = rsp_exp[idx_start:]
    rsp_deriv_sliced = rsp_deriv[idx_start:]

    # Initial baselines from ends (low T ~ ds; high T ~ ss)
    n = len(temp_sliced)

    # take the first and last 10% of data points for initial baselines
    k_end = int(round(fraction_init * n))
    m_ds_init, b_ds_init = np.polyfit(temp_sliced[:k_end], rsp_sliced[:k_end], deg=1)
    m_ss_init, b_ss_init = np.polyfit(temp_sliced[n - k_end:], rsp_sliced[n - k_end:], deg=1)

    # Initial thermodynamic parameters (association: ΔH<0, ΔS<0)
    delta_h_init = -200.0e3  # J/mol
    delta_s_init = -600.0   # J/mol/K

    # Fitting directly to the response curve----------------------------
    # Initial parameters:
    params_init = np.asarray([ delta_h_init, delta_s_init, b_ds_init, m_ds_init, b_ss_init, m_ss_init], dtype=float)

    res = optimize.minimize(merit_rsp,
                            params_init,
                            args=(temp_sliced, conc_ds, rsp_sliced),
                            method="Nelder-Mead",
                            options=nm_options
                            )

    # Return the fitted parameters
    params_rsp = params_from_array(res.x)
    rsp_fit_sliced = calc_rsp(temp_sliced, conc_ds, params_rsp)
    rss_direct = float(np.nansum((rsp_fit_sliced - rsp_sliced) ** 2))

    # Tm from fitted dH, dS, and C
    tm_anal_direct = np.round(calc_tm_anal(conc_ds, params_rsp), decimals=1)
    tm_anal_direct_celsius = np.round(tm_anal_direct -273.15, decimals=1)

    # ----- Pad fitted/residual back to full length with NaNs so columns align -----
    rsp_fit_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    residual_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    rsp_fit_full[idx_start:] = rsp_fit_sliced
    residual_full[idx_start:] = (rsp_fit_sliced - rsp_sliced)

    # TODO: Baselines (for diagnostic purpose)
    baseline_ds = params_rsp.b_ds + params_rsp.m_ds*temp_kelvin
    baseline_ss = params_rsp.b_ss + params_rsp.m_ss*temp_kelvin

    # Fitting to the first derivative curve-----------------------------
    params_init_deriv = res.x.copy() # Take the parameters

    res_deriv = optimize.minimize(merit_rsp_deriv,
                                  params_init_deriv,
                                  args=(temp_sliced, conc_ds, rsp_deriv_sliced),
                                  method="Nelder-Mead",
                                  options=nm_options
                                  )

    # Return the fitted parameters (derivative fit)
    params_deriv = params_from_array(res_deriv.x)
    rsp_deriv_fit_sliced = calc_rsp_deriv(temp_sliced, conc_ds, params_deriv)
    rss_deriv = float(np.nansum((rsp_deriv_fit_sliced - rsp_deriv_sliced) ** 2))

    # Tm from fitted dH, dS, and C (derivative fit)
    tm_anal_deriv = np.round(calc_tm_anal(conc_ds, params_deriv), decimals=1)
    tm_anal_deriv_celsius = np.round(tm_anal_deriv -273.15, decimals=1)

    # Pad fitted/residual back to full length with NaNs so columns align
    rsp_deriv_fit_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    residual_deriv_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    rsp_deriv_fit_full[idx_start:] = rsp_deriv_fit_sliced
    residual_deriv_full[idx_start:] = (rsp_deriv_fit_sliced - rsp_deriv_sliced)

    # Assemble wide outputs
    curves_out[f"{well_ID[i]}_exp"] = rsp_exp
    curves_out[f"{well_ID[i]}_fitted"]   = rsp_fit_full
    curves_out[f"{well_ID[i]}_residual"] = residual_full
    curves_out[f"{well_ID[i]}_baseline_ds"] = baseline_ds
    curves_out[f"{well_ID[i]}_baseline_ss"] = baseline_ss
    curves_out[f"{well_ID[i]}_deriv"] = rsp_deriv
    curves_out[f"{well_ID[i]}_deriv_fitted"] = rsp_deriv_fit_full
    curves_out[f"{well_ID[i]}_deriv_residual"] = residual_deriv_full

    # Parameter row
    params_rows.append({
        "curve_id": str(well_ID[i]),
        "DeltaH_anal_direct (kJ/mol)": float(np.round(params_rsp.delta_h / 1000, decimals=1)),
        "DeltaS_anal_direct (J/mol/K)": float(np.round(params_rsp.delta_s, decimals=1)),
        "Tm_anal_direct (C)": tm_anal_direct_celsius if np.isfinite(tm_anal_direct) else np.nan,
        "RSS_direct": rss_direct,
        "b_ds": params_rsp.b_ds,
        "m_ds": params_rsp.m_ds,
        "b_ss": params_rsp.b_ss,
        "m_ss": params_rsp.m_ss,
        "DeltaH_anal_deriv (kJ/mol)": float(np.round(params_deriv.delta_h / 1000, decimals=1)),
        "DeltaS_anal_deriv (J/mol/K)": float(np.round(params_deriv.delta_s, decimals=1)),
        "Tm_anal_deriv (C)": tm_anal_deriv_celsius if np.isfinite(tm_anal_deriv_celsius) else np.nan,
        "RSS_deriv": rss_deriv,
        "Tm_estim (C)": float(np.round(tm_estim - 273.15, decimals=1)) if np.isfinite(tm_estim) else np.nan,
        })

    # Output the plots--------------------------------------------------
    if "uv" in method.casefold():
        ylabel = "Absorbance (AU)"
        ylabel_short = "AU"
    elif "qpcr" in method.casefold():
        ylabel = "Fluorescence (RFU)"
        ylabel_short = "RFU"
    else:
        ylabel = "Response"
        ylabel_short = "Response"

    # Response fit figure
    out_png_resp = os.path.join(fig_dir,
                                f"{file_base}_{well_ID[i]}_response_fit.png"
                                )

    plot_overlay(temp_c=temp_celsius,
                 y_exp=rsp_exp,
                 y_fit=rsp_fit_full,
                 residual=residual_full,
                 title=f"{well_ID[i]}: Direct Fit",
                 ylabel=ylabel,
                 method=method,
                 data_type="direct",
                 out_path=out_png_resp
                 )

    # Derivative fit figure
    out_png_deriv = os.path.join(fig_dir,
                                 f"{file_base}_{well_ID[i]}_derivative_fit.png"
                                 )

    plot_overlay(temp_c=temp_celsius,
                 y_exp=rsp_deriv,
                 y_fit=rsp_deriv_fit_full,
                 residual=residual_deriv_full,
                 title=f"{well_ID[i]}: Derivative Fit ",
                 ylabel=f"d{ylabel_short}/dT",
                 method=method,
                 data_type="derivative",
                 out_path=out_png_deriv
                 )

# ---------- Write outputs ----------
curves_df  = pd.DataFrame(curves_out)
params_df  = pd.DataFrame(params_rows,
                          columns=["curve_id","DeltaH_anal_direct (kJ/mol)",
                                   "DeltaS_anal_direct (J/mol/K)","Tm_anal_direct (C)",
                                   "RSS_direct","b_ds","m_ds","b_ss","m_ss",
                                   "DeltaH_anal_deriv (kJ/mol)","DeltaS_anal_deriv (J/mol/K)",
                                   "Tm_anal_deriv (C)","RSS_deriv","Tm_estim (C)"
                                   ])

curves_out_path = os.path.join(dir_name, f"{file_base}_fitted_curves_{method}.csv")
params_out_path = os.path.join(dir_name, f"{file_base}_fitted_parameters_{method}.csv")

curves_df.to_csv(curves_out_path, index=False)
params_df.to_csv(params_out_path, index=False)

print("Done.")
print("Curves:     ", curves_out_path)
print("Parameters: ", params_out_path)
print("Figures:    ", fig_dir)
