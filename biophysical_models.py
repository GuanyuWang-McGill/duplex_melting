# This file stores the functions of the biophysical models
from dataclasses import dataclass
import numpy as np

# Global constants------------------------------------------------------
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

# Biophysical Models ---------------------------------------------------
def calc_ka(temp: np.ndarray, params: ThermoParams) -> np.ndarray:
    """
    Calculate the association constant Ka
        For intramolecular duplex hybridization A <-> C, Ka = [C]/[A]
        For bimolecular self-complementary duplex hybridization A + A <-> C, Ka = [C]/[A]^
        For bimolecular duplex hybridization A + B <-> C, Ka = [C]/[A][B]
    :return: equilibrium constant in mol-1
    """
    return np.exp(params.delta_s / R - params.delta_h / (R * temp))


def calc_frac_fold(temp: np.ndarray,
                   conc: float,
                   params: ThermoParams,
                   is_intramolecular = False,
                   is_self_complementary = False
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
             is_intramolecular = False,
             is_self_complementary = False
             ) -> np.ndarray:
    """
    Take the `temperature`, `concentration`, and `params` to simulate the response
    of the machine (RFU or Abs).
    params = [ΔH, ΔS, b_ds, m_ds, b_ss, m_ss]
    :return: simulated response
    """
    frac_fold = calc_frac_fold(temp, conc, params, is_intramolecular, is_self_complementary)

    return (params.b_ds + params.m_ds * temp + (1- frac_fold)
            * (params.b_ss - params.b_ds + (params.m_ss - params.m_ds) * temp))


def calc_rsp_deriv(temp: np.ndarray,
                   conc: float,
                   params: ThermoParams,
                   is_intramolecular = False,
                   is_self_complementary = False
                   ) -> np.ndarray:
    """
    Take the `temperature` and `params` to simulate the first derivative
    of response of the machine (RFU or Abs) of duplex melting
    :return: derivative of the response with respect to T
    """
    frac_fold = calc_frac_fold(temp, conc, params, is_intramolecular, is_self_complementary)

    ka = calc_ka(temp, params)

    return (params.m_ds + (params.m_ss - params.m_ds) * (1 - frac_fold)
            - ((params.b_ss - params.b_ds) + (params.m_ss - params.m_ds) * temp)
            * (params.delta_h / (R * temp ** 2))
            * (frac_fold / np.sqrt(1 + 4 * ka * conc )))


def calc_tm_anal(conc: float,
                 params: ThermoParams,
                 is_intramolecular=False,
                 is_self_complementary=False
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


def calc_rsp_quad(temp: np.ndarray,
                       conc: float,
                       params: ThermoParams,
                       is_intramolecular = False,
                       is_self_complementary = False
                       ) -> np.ndarray:
    """
    Take the `temperature`, `concentration`, and `params` to simulate the response
    of the machine (RFU or Abs).
    params = [ΔH, ΔS, b_ds, m_ds, b_ss, m_ss]
    :return: simulated response
    """
    frac_fold = calc_frac_fold(temp, conc, params, is_intramolecular, is_self_complementary)

    # This is the quadratic baseline
    return (params.b_ds + params.m_ds * temp ** 2 + (1- frac_fold)
            * (params.b_ss - params.b_ds + (params.m_ss - params.m_ds * temp) * temp))

def calc_rsp_deriv_quad(temp: np.ndarray,
                   conc: float,
                   params: ThermoParams,
                   is_intramolecular = False,
                   is_self_complementary = False
                   ) -> np.ndarray:
    """
    Take the `temperature` and `params` to simulate the first derivative
    of response of the machine (RFU or Abs) of duplex melting
    :return: derivative of the response with respect to T
    """
    frac_fold = calc_frac_fold(temp, conc, params, is_intramolecular, is_self_complementary)

    ka = calc_ka(temp, params)

    # This function uses a quadratic baseline
    return (2* params.m_ds * temp + (params.m_ss - 2 * params.m_ds * temp) * (1 - frac_fold)
            - ((params.b_ss - params.b_ds) + (params.m_ss - 2 * params.m_ds * temp) * temp)
            * (params.delta_h / (R * temp ** 2))
            * (frac_fold / np.sqrt(1 + 4 * ka * conc)))