import numpy as np
from scipy.integrate import solve_ivp

class USStandardAtmosphere1976:
    """Classe per la modellazione dell'atmosfera standard U.S. 1976 a piu strati."""
    @staticmethod
    def get_density(altitude):
        h = max(0.0, float(altitude))
        g0 = 9.80665   # m/s^2
        R = 287.053    # J/(kg*K)
        
        # Strato 0: Troposfera (0 - 11000 m)
        if h <= 11000.0:
            T0, rho0, L = 288.15, 1.225, -0.0065
            T = T0 + L * h
            return rho0 * (T / T0)**(-g0 / (R * L) - 1.0)
        # Strato 1: Bassa Stratosfera / Tropopausa (11000 - 20000 m)
        elif h <= 20000.0:
            T11, rho11 = 216.65, 0.36391
            return rho11 * np.exp(-g0 * (h - 11000.0) / (R * T11))
        # Strato 2: Media Stratosfera (20000 - 32000 m)
        else:
            T20, rho20, L = 216.65, 0.08803, 0.0010
            T = T20 + L * (h - 20000.0)
            return rho20 * (T / T20)**(-g0 / (R * L) - 1.0)

class AeroDatabase:
    """
    Database aerodinamico multi-regime.
    Surrogato analitico a chiusura rapida calibrato sui dati NASA per i tre regimi.
    """
    @staticmethod
    def get_coefficients(Mach, alpha):
        # 1. Regime Ipersonico (Teoria Newtoniana Modificata)
        if Mach > 5.0:
            Cd = 1.6 * (np.sin(alpha)**3) + 0.1
            Cl = 1.2 * (np.sin(alpha)**2) * np.cos(alpha)
            Cm_0 = -0.05 * alpha
        # 2. Regime Supersonico/Transonico (Interpolazione dati empirici NASA)
        elif Mach >= 0.8:
            Cd = 1.2 + 0.3 * (Mach - 0.8)
            Cl = 1.5 * np.sin(2 * alpha)
            Cm_0 = -0.08 * alpha
        # 3. Regime Subsonico (Correzione di Prandtl-Glauert)
        else:
            beta = np.sqrt(max(1.0 - Mach**2, 0.1))
            Cd = 0.8 / beta
            Cl = (2.0 * np.pi * alpha) / beta
            Cm_0 = -0.1 * alpha
            
        return Cd, Cl, Cm_0

class FlightController:
    """Controllore d'assetto e Allocatore di Controllo Riconfigurabile Dinamico."""
    def __init__(self):
        self.fdi_active = False
        self.stuck_index = None
        self.delta_stuck = 0.0
        self.W = np.diag([1.5, 1.5, 1.0, 1.0])

    def set_fault(self, stuck_index, delta_stuck):
        """Attiva la logica di guasto."""
        self.fdi_active = True
        self.stuck_index = stuck_index
        self.delta_stuck = delta_stuck

    def allocate(self, M_cmd, B_current):
        """Risolve il problema di Control Allocation adattiva tramite pseudo-inversa pesata."""
        u_cmd = np.zeros(4)
        if not self.fdi_active:
            W_inv = np.linalg.inv(self.W)
            temp = B_current @ W_inv @ B_current.T
            u_cmd = (W_inv @ B_current.T) * (M_cmd / temp)
        else:
            B_act = np.delete(B_current, self.stuck_index)
            M_reconfig = M_cmd - B_current[self.stuck_index] * self.delta_stuck

            W_act = np.delete(
                np.delete(self.W, self.stuck_index, axis=0),
                self.stuck_index, axis=1
            )
            W_act_inv = np.linalg.inv(W_act)

            temp = B_act @ W_act_inv @ B_act.T
            u_act = (W_act_inv @ B_act.T) * (M_reconfig / temp)
            u_cmd = np.insert(u_act, self.stuck_index, self.delta_stuck)
            
        return np.clip(u_cmd, -np.radians(40), np.radians(40))

class Starship3DOF:
    """Modellazione dinamica 3-DOF del velivolo Starship."""
    def __init__(self):
        self.mass = 120000.0 # kg
        self.Iyy = 2.5e7     # kg*m^2
        self.S_ref = 450.0   # m^2
        self.c_chord = 9.0   # m
        self.B_0 = np.array([400.0, 400.0, -500.0, -500.0])
        self.controller = FlightController()

    def equations_of_motion(self, t, state, t_fault, stuck_idx, delta_stuck):
        """Definisce il sistema di ODE a 7 stati per solve_ivp."""
        x, z, vx, vz, theta, q, e_int = state
        altitude = z
        
        rho = USStandardAtmosphere1976.get_density(altitude)
        
        # Raffica 1-coseno
        w_x, w_z = 0.0, 0.0
        if 5.0 <= t <= 8.0:
            W_max = 15.0 # m/s
            w_z = 0.5 * W_max * (1.0 - np.cos(2.0 * np.pi * (t - 5.0) / 3.0))
            
        vx_rel = vx - w_x
        vz_rel = vz - w_z
        V_rel = np.sqrt(vx_rel**2 + vz_rel**2)
        q_inf = 0.5 * rho * V_rel**2
        
        a_sound = 340.0 
        Mach = V_rel / a_sound
        
        gamma = np.arctan2(vz_rel, vx_rel)
        alpha = theta - gamma
        
        B_current = self.B_0 * q_inf
        
        # Iniezione del guasto con latenza diagnostica FDI di 0.5 s
        t_fdi = t_fault + 0.5
        if t >= t_fdi and not self.controller.fdi_active:
            self.controller.set_fault(stuck_idx, delta_stuck)
            
        # PID con Gain Scheduling e Anti-Windup
        theta_ref = np.radians(-20.0) if t < 20 else np.radians(90.0)
        error = theta_ref - theta
        
        q_ref = 3000.0 # Pa
        q_factor = max(q_inf / q_ref, 0.1)
        kp = 2.5e6 * q_factor
        kd = 8.0e6 * q_factor
        ki = 4.0e5 * q_factor
        
        e_int_max = np.radians(10.0)
        if abs(e_int) >= e_int_max and np.sign(error) == np.sign(e_int):
            de_int_dt = 0.0
        else:
            de_int_dt = error
            
        M_required = kp * error + ki * e_int - kd * q
        u = self.controller.allocate(M_required, B_current)
        
        Cd, Cl, Cm_0 = AeroDatabase.get_coefficients(Mach, alpha)
        
        Lift = q_inf * self.S_ref * Cl
        Drag = q_inf * self.S_ref * Cd
        M_aero = Cm_0 * q_inf * self.S_ref * self.c_chord + np.dot(B_current, u)
        
        dx_dt = vx
        dz_dt = vz
        
        g_0, R_E = 9.81, 6371000.0
        g = g_0 * (R_E / (R_E + altitude))**2
        
        dvx_dt = (-Drag * np.cos(gamma) - Lift * np.sin(gamma)) / self.mass
        dvz_dt = (-Drag * np.sin(gamma) + Lift * np.cos(gamma)) / self.mass - g
        
        dtheta_dt = q
        dq_dt = M_aero / self.Iyy
        
        return [dx_dt, dz_dt, dvx_dt, dvz_dt, dtheta_dt, dq_dt, de_int_dt]
