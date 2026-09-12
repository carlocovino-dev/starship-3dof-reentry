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
        Alpha in ingresso in radianti -> Convertito in gradi per l'interpolatore.
        """
        point = np.array([[Mach, np.degrees(alpha)]])
        C_L = float(self.cl_interp(point)[0])
        C_D = float(self.cd_interp(point)[0])
        C_m = float(self.cm_interp(point)[0])
        return C_L, C_D, C_m


# ==============================================================================
# BLOCCO DI TEST PER L'ESECUZIONE
# ==============================================================================
if __name__ == '__main__':
    print("==================================================================")
    print(" TEST INTERPOLATORE AERODINAMICO (RegularGridInterpolator)")
    print("==================================================================")

    # 1. Definizione delle griglie di input per il database (es. Mach e Alpha)
    mach_grid = np.array([0.2, 0.8, 1.2, 2.5, 5.0, 10.0])              # Griglia Mach 1D
    alpha_grid = np.linspace(-20.0, 90.0, 12)                          # Griglia Alpha in gradi 1D

    # 2. Creazione di matrici dati di test (Griglia 2D [Mach x Alpha])
    M_mesh, A_mesh = np.meshgrid(mach_grid, alpha_grid, indexing='ij')
    
    cl_data = 1.2 * np.sin(2 * np.radians(A_mesh))
    cd_data = 0.1 + 1.5 * (np.sin(np.radians(A_mesh))**2)
    cm_data = -0.05 * np.radians(A_mesh)

    # 3. Istanziazione della classe AeroDatabase
    aero_db = AeroDatabase(mach_grid, alpha_grid, cl_data, cd_data, cm_data)

    # 4. Valutazione di test (es. Mach = 2.0 e Alpha = +15 gradi in radianti)
    mach_test = 2.0
    alpha_test_rad = np.radians(15.0)

    cl, cd, cm = aero_db.evaluate(mach_test, alpha_test_rad)

    print(f"\nCondizioni di Test:")
    print(f"- Mach: {mach_test:.2f}")
    print(f"- Alpha: {np.degrees(alpha_test_rad):.2f}° ({alpha_test_rad:.4f} rad)")
    print(f"\nCoefficienti Interpolati:")
    print(f"- C_L (Portanza) : {cl:.4f}")
    print(f"- C_D (Resistenza)    : {cd:.4f}")
    print(f"- C_m (Beccheggio)    : {cm:.4f}")
