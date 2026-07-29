import numpy as np
from scipy.interpolate import RegularGridInterpolator

class AeroDatabase:
    def __init__(self, mach_grid, alpha_grid, cl_data, cd_data, cm_data):
        """
        Inizializzazione degli interpolatori per i coefficienti aerodinamici.
        Grid: Mach (1D), Alpha in gradi (1D).
        Data arrays: Matrici 2D [Mach, Alpha].
        """
        self.cl_interp = RegularGridInterpolator((mach_grid, alpha_grid), cl_data, bounds_error=False, fill_value=None)
        self.cd_interp = RegularGridInterpolator((mach_grid, alpha_grid), cd_data, bounds_error=False, fill_value=None)
        self.cm_interp = RegularGridInterpolator((mach_grid, alpha_grid), cm_data, bounds_error=False, fill_value=None)
        
    def evaluate(self, Mach, alpha):
        """
        Calcolo dei coefficienti aerodinamici all'istante di integrazione corrente.
        """
        point = np.array([Mach, np.degrees(alpha)])
        C_L = float(self.cl_interp(point))
        C_D = float(self.cd_interp(point))
        C_m = float(self.cm_interp(point))
        return C_L, C_D, C_m
