import numpy as np
from scipy.interpolate import RegularGridInterpolator


class AeroDatabase:
    def __init__(self, mach_grid, alpha_grid, cl_data, cd_data, cm_data):
        self.cl_interp = RegularGridInterpolator((mach_grid, alpha_grid), cl_data, bounds_error=False, fill_value=None)
        self.cd_interp = RegularGridInterpolator((mach_grid, alpha_grid), cd_data, bounds_error=False, fill_value=None)
        self.cm_interp = RegularGridInterpolator((mach_grid, alpha_grid), cm_data, bounds_error=False, fill_value=None)

    def evaluate(self, Mach, alpha_rad):
        M_arr = np.asarray(Mach)
        a_deg = np.degrees(alpha_rad)
        M_b, a_b = np.broadcast_arrays(M_arr, a_deg)
        points = np.column_stack((M_b.ravel(), a_b.ravel()))

        C_L = self.cl_interp(points).reshape(M_b.shape)
        C_D = self.cd_interp(points).reshape(M_b.shape)
        C_m = self.cm_interp(points).reshape(M_b.shape)

        if C_L.ndim == 0:
            return float(C_L), float(C_D), float(C_m)
        return C_L, C_D, C_m


# ==============================================================================
# 1. COSTRUZIONE DATABASE SUI VALORI NOMINALI DI TABELLA 1.1
# ==============================================================================
mach_grid = np.array([0.1, 0.4, 0.75, 0.8, 1.0, 1.2, 1.3, 2.5, 4.8, 5.0, 7.0, 10.0])
alpha_grid = np.arange(50.0, 90.0, 2.0)  # Griglia attorno al belly flop

# Nodi a alpha = 70° conformi ai range della Tabella 1.1:
# Sub:   CD in [1.20, 1.35], CL in [0.30, 0.45], Cm = -0.05
# Trans: CD in [1.35, 1.60], CL in [0.25, 0.40], Cm non lineare
# Super: CD in [1.50, 1.75], CL in [0.20, 0.35], Cm = -0.08
# Iper:  CD in [1.70, 1.84], CL in [0.15, 0.25], Cm = -0.10
cd_at_70 = np.array([1.22, 1.26, 1.33, 1.37, 1.58, 1.52, 1.55, 1.66, 1.73, 1.72, 1.78, 1.82])
cl_at_70 = np.array([0.42, 0.38, 0.32, 0.31, 0.28, 0.26, 0.27, 0.24, 0.21, 0.21, 0.18, 0.16])
cm_at_70 = np.array([-0.05, -0.05, -0.05, -0.055, -0.065, -0.075, -0.078, -0.080, -0.082, -0.098, -0.100, -0.100])

a_rad = np.radians(alpha_grid)
ref_rad = np.radians(70.0)

# Funzioni di cross-flow normalizzate
g_D = (np.sin(a_rad)**3) / (np.sin(ref_rad)**3)
g_L = (np.sin(a_rad)**2 * np.cos(a_rad)) / (np.sin(ref_rad)**2 * np.cos(ref_rad))
g_m = np.sin(a_rad) / np.sin(ref_rad)

cd_data = np.outer(cd_at_70, g_D)
cl_data = np.outer(cl_at_70, g_L)
cm_data = np.outer(cm_at_70, g_m)

aero_db = AeroDatabase(mach_grid, alpha_grid, cl_data, cd_data, cm_data)


# ==============================================================================
# 2. GENERAZIONE DI 10 PUNTI PER CIASCUN REGIME E TABULAZIONE
# ==============================================================================
def identifica_regime(M):
    if M < 0.8:
        return "Subsonico"
    elif 0.8 <= M <= 1.2:
        return "Transonico"
    elif 1.2 < M <= 5.0:
        return "Supersonico"
    else:
        return "Ipersonico"


# 10 campioni per ciascun regime con alpha attorno alla condizione nominale di 70°
campioni_regimi = {
    "Subsonico":   (np.linspace(0.15, 0.75, 10), np.linspace(69.0, 71.0, 10)),
    "Transonico":  (np.linspace(0.80, 1.20, 10), np.linspace(69.5, 70.5, 10)),
    "Supersonico": (np.linspace(1.30, 4.80, 10), np.linspace(68.5, 71.5, 10)),
    "Ipersonico":  (np.linspace(5.20, 9.50, 10), np.linspace(69.0, 71.0, 10)),
}

print(f"{'#':<3} | {'Mach':<6} | {'Alpha (deg)':<11} | {'C_D':<7} | {'C_L':<7} | {'C_m':<8} | {'Regime di Volo'}")
print("-" * 65)

contatore = 1
for regime_nome, (m_vals, a_vals) in campioni_regimi.items():
    cls, cds, cms = aero_db.evaluate(m_vals, np.radians(a_vals))
    for i in range(10):
        reg_calcolato = identifica_regime(m_vals[i])
        print(f"{contatore:<3} | {m_vals[i]:<6.3f} | {a_vals[i]:<11.2f} | {cds[i]:<7.4f} | {cls[i]:<7.4f} | {cms[i]:<8.4f} | {reg_calcolato}")
        contatore += 1
    print("-" * 65)
