# Fit melt curves directly on RFU or absorbance with a 2-state model,
# give out DeltaH, DeltaS and Tm
#
# INPUT CSV: first row is headers. One header is "Temperature"
# It should be from lower temperature to higher temperature
# All other numeric columns are treated as sample wells
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
import os
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from scipy import optimize

from biophysical_models import (ThermoParams,
                                calc_tm_anal,
                                calc_tm_estimate,
                                calc_rsp,
                                calc_rsp_deriv
                                )
from fitting import (merit_rsp,
                     merit_rsp_deriv,
                     params_from_array
                     )
from plotting import plot_overlay

# File Handling---------------------------------------------------------
# Open a CSV file-------------------------------------------------------
root = Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    filetypes=[("CSV files", "*.csv")],
    title="Select a file containing melting curves"
)
root.destroy()

if not file_path:
    print("No file selected!")
    exit()

# Example files
# file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_EG_2.5uM.csv"
# file_path = "/Users/guanyuwang/IdeaProjects/Fitting_algorithms/fit_duplex_melting/T-NP_UV_2.5uM.csv"

# Experiment settings --------------------------------------------------
conc_ds               = 2.5e-6   # strand concentration (M) for bimolecular models; ignored if intramolecular
is_intramolecular     = False    # True for hairpins (unimolecular)
is_self_complementary = False    # True for self-complementary duplexes (homodimer)
method                = str(input("What is the method used for the measurement (UV or qPCR): "))

# Fitting settings -----------------------------------------------------
is_truncate           = False    # truncate only for qPCR data
temp_truncate         = 30       # Truncate the data at Tm - temp_truncate
fraction_init         = 0.10     # fraction (10%) of points at each end for baseline initial guesses
nm_options            = dict(maxiter=1e6,
                             maxfev=1e6,
                             xatol=1e-6,
                             fatol=1e-6
                             )   # Nelder-Mead minimization options

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
    temp_sliced_celsius = temp_sliced - 273.15 # This is for plotting
    rsp_sliced = rsp_exp[idx_start:]
    rsp_deriv_sliced = rsp_deriv[idx_start:]

    # Initial baselines from ends (low T ~ ds; high T ~ ss)
    n = len(temp_sliced)

    # Take the first and last `fraction_init`% of data points for initial baselines
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

    # Pad fitted/residual back to full length with NaNs so columns align
    rsp_fit_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    residual_full = np.full_like(temp_kelvin, np.nan, dtype=float)
    rsp_fit_full[idx_start:] = rsp_fit_sliced
    residual_sliced = rsp_fit_sliced - rsp_sliced
    residual_full[idx_start:] = residual_sliced

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
    residual_deriv_sliced = rsp_deriv_fit_sliced - rsp_deriv_sliced
    residual_deriv_full[idx_start:] = residual_deriv_sliced

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
        "Tm_estim (C)": float(np.round(tm_estim - 273.15, decimals=0)) if np.isfinite(tm_estim) else np.nan,
    })

    # Assemble wide outputs
    curves_out[f"{well_ID[i]}_exp"] = rsp_exp
    curves_out[f"{well_ID[i]}_fitted"]   = rsp_fit_full
    curves_out[f"{well_ID[i]}_residual"] = residual_full
    # TODO: Baselines (for diagnostic purpose)
    curves_out[f"{well_ID[i]}_baseline_ds"] = baseline_ds
    curves_out[f"{well_ID[i]}_baseline_ss"] = baseline_ss
    curves_out[f"{well_ID[i]}_deriv"] = rsp_deriv
    curves_out[f"{well_ID[i]}_deriv_fitted"] = rsp_deriv_fit_full
    curves_out[f"{well_ID[i]}_deriv_residual"] = residual_deriv_full
    ####################################################################

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
                                f"{file_base}_{well_ID[i]}_{method}_direct_fit.png"
                                )

    plot_overlay(temp_c=temp_sliced_celsius,
                 y_exp=rsp_sliced,
                 y_fit=rsp_fit_sliced,
                 residual=residual_sliced,
                 title=f"{well_ID[i]}: Direct Fit",
                 ylabel=ylabel,
                 method=method,
                 data_type="direct",
                 out_path=out_png_resp
                 )

    # Derivative fit figure
    out_png_deriv = os.path.join(fig_dir,
                                 f"{file_base}_{well_ID[i]}_{method}_derivative_fit.png"
                                 )

    plot_overlay(temp_c=temp_sliced_celsius,
                 y_exp=rsp_deriv_sliced,
                 y_fit=rsp_deriv_fit_sliced,
                 residual=residual_deriv_sliced,
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