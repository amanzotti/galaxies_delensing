from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys
import configparser as ConfigParser
import numpy as np
from joblib import Parallel, delayed
import scipy.integrate as integrate
import profiling
from profiling.sampling import SamplingProfiler

if __name__ == "__main__":
    import rho_to_Bres
    # cbb_der = rho_to_Bres.compute_deriv_grid(delta_phi=30,delta_b=30)
    cbb_der = rho_to_Bres.compute_deriv_grid_CEE(delta_e=30, delta_b=30)
