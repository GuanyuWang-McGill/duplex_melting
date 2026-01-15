# This file stores the merit functions and other helpers
import numpy as np

from biophysical_models import (ThermoParams,
                                calc_rsp,
                                calc_rsp_deriv
                                )

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