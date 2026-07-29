import numpy as np


class Atmosphere1976:
    """Modello US Standard Atmosphere 1976 esteso a piu strati."""
    R = 287.05      # Costante specifica dell'aria secca [J/(kg*K)]
    g = 9.80665     # Accelerazione di gravita standard [m/s^2]
    gamma = 1.4     # Rapporto dei calori specifici

    @staticmethod
    def get_properties(alt):
        """Restituisce densita [kg/m^3] e velocita del suono [m/s]."""
        z = max(0.0, alt)
        if z < 11000.0:  # Troposfera
            T = 288.15 - 0.0065 * z
            rho = 1.225 * (T / 288.15) ** ((Atmosphere1976.g / (0.0065 * Atmosphere1976.R)) - 1.0)
        elif z < 20000.0:  # Bassa Stratosfera (Isoterma)
            T = 216.65
            rho_11 = 0.36391
            rho = rho_11 * np.exp(-Atmosphere1976.g * (z - 11000.0) / (Atmosphere1976.R * T))
        else:  # Alta Stratosfera (fino a 80 km)
            T = 216.65 + 0.001 * (z - 20000.0)
            rho = 0.08803 * (T / 216.65) ** (- (Atmosphere1976.g / (0.001 * Atmosphere1976.R)) - 1.0)
            
        a = np.sqrt(Atmosphere1976.gamma * Atmosphere1976.R * T)
        return rho, a


class Starship3DOF:
    """Rappresentazione fisica e dinamica del veicolo Starship."""
    def __init__(self):
        self.m = 120000.0       # Massa a secco [kg]
        self.Iyy = 2.5e7        # Inerzia di beccheggio [kg*m^2]
        self.S_ref = 450.0      # Superficie di riferimento [m^2]
        self.c_ref = 9.0        # Corda di riferimento [m]
        self.g = 9.80665

    def dynamics_derivatives(self, t, state, delta_FL, delta_FR, delta_AL, delta_AR, aero_model):
        """
        Calcola il vettore delle derivate dello stato [vx, vz, ax, az, q, q_dot].
        state: [x_pos, z_pos, x_vel, z_vel, theta_rad, q_rad_s]
        """
        x, z, vx, vz, theta, q = state
        V = np.sqrt(vx**2 + vz**2)
        
        # Gestione della singolarita per velocita nulle
        if V < 1e-3:
            gamma = -np.pi / 2.0
            V = 1e-3
        else:
            gamma = np.arctan2(vz, vx)

        alpha = theta - gamma
        rho, a = Atmosphere1976.get_properties(z)
        q_inf = 0.5 * rho * V**2  # Pressione dinamica
        
        # Estrazione coefficienti dal modello aerodinamico su 4 attuatori
        CL, CD, Cm = aero_model.get_coefficients(alpha, V / a, delta_FL, delta_FR, delta_AL, delta_AR)
        
        # Forze nel sistema di riferimento vento
        L = q_inf * self.S_ref * CL
        D = q_inf * self.S_ref * CD
        
        # Proiezione nel sistema di riferimento inerziale
        Fx = -D * np.cos(gamma) - L * np.sin(gamma)
        Fz = -D * np.sin(gamma) + L * np.cos(gamma)
        
        # Momento aerodinamico rispetto al centro di massa
        My = q_inf * self.S_ref * self.c_ref * Cm
        
        # Equazioni cardinali della dinamica
        ax = Fx / self.m
        az = (Fz / self.m) - self.g
        q_dot = My / self.Iyy
        
        return [vx, vz, ax, az, q, q_dot]


# Modello aerodinamico di test per la valutazione della dinamica
class MockAeroModel:
    @staticmethod
    def get_coefficients(alpha, Mach, d_FL, d_FR, d_AL, d_AR):
        CL = 0.8 * np.sin(2 * alpha)
        CD = 1.0 * (np.sin(alpha)**2) + 0.1
        Cm = -0.1 * np.sin(alpha) + 0.05 * (d_FL + d_FR - d_AL - d_AR)
        return CL, CD, Cm


# ==============================================================================
# BLOCCO DI TEST PER L'ESECUZIONE
# ==============================================================================
if __name__ == '__main__':
    print("==================================================================")
    print(" TEST ATMOSFERA 1976 E DINAMICA STARSHIP 3-DOF")
    print("==================================================================")

    # 1. Test modello Atmosfera a 15 km
    z_test = 15000.0
    rho, a = Atmosphere1976.get_properties(z_test)
    print(f"\n[Atmosphere1976] Proprietà a z = {z_test/1000:.1f} km:")
    print(f"  - Densità aria (rho)   : {rho:.4f} kg/m^3")
    print(f"  - Velocità suono (a)   : {a:.2f} m/s")

    # 2. Test calcolo derivate dinamiche
    ship = Starship3DOF()
    aero_model = MockAeroModel()

    # Stato: [x=0m, z=15000m, vx=120m/s, vz=-70m/s, theta=-20deg, q=0rad/s]
    state0 = [0.0, 15000.0, 120.0, -70.0, np.radians(-20.0), 0.0]
    
    # Deflessioni flap di test (in radianti)
    d_FL = np.radians(5.0)
    d_FR = np.radians(5.0)
    d_AL = np.radians(-10.0)
    d_AR = np.radians(-10.0)

    derivatives = ship.dynamics_derivatives(0.0, state0, d_FL, d_FR, d_AL, d_AR, aero_model)

    print(f"\n[Starship3DOF] Derivate dello stato calcolate a t = 0 s:")
    print(f"  - Velocità x (vx)          : {derivatives[0]:+8.2f} m/s")
    print(f"  - Velocità z (vz)          : {derivatives[1]:+8.2f} m/s")
    print(f"  - Accelerazione x (ax)     : {derivatives[2]:+8.2f} m/s^2")
    print(f"  - Accelerazione z (az)     : {derivatives[3]:+8.2f} m/s^2")
    print(f"  - Velocità angolare (q)    : {derivatives[4]:+8.2f} rad/s")
    print(f"  - Acc. angolare (q_dot)    : {derivatives[5]:+8.4f} rad/s^2")
