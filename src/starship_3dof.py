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
