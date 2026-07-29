import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def get_theta_ref(t):
    """Calcola il riferimento d'assetto theta_ref in radianti in funzione del tempo."""
    if t < 20.0:
        return np.radians(-20.0)
    elif t < 28.0:
        p = (t - 20.0) / 8.0
        smooth_factor = 3 * p**2 - 2 * p**3
        return np.radians(-20.0 + 110.0 * smooth_factor)
    else:
        return np.radians(90.0)


class USStandardAtmosphere1976:
    """Classe per la modellazione dell'atmosfera standard U.S. 1976 a più strati."""
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
    Database aerodinamico multi-regime continuo.
    Formulazione continua per angoli d'attacco estesi (0-360 deg) durante la manovra di flip.
    """
    @staticmethod
    def get_coefficients(Mach, alpha):
        if Mach > 5.0:
            Cd = 1.6 * (np.sin(alpha)**2) + 0.1
            Cl = 1.2 * np.sin(2 * alpha)
            Cm_0 = -0.05 * np.sin(alpha)
        elif Mach >= 0.8:
            Cd = 1.2 * (np.sin(alpha)**2) + 0.15
            Cl = 1.1 * np.sin(2 * alpha)
            Cm_0 = -0.08 * np.sin(alpha)
        else:
            Cd = 1.0 * (np.sin(alpha)**2) + 0.1
            Cl = 0.8 * np.sin(2 * alpha)
            Cm_0 = -0.1 * np.sin(alpha)
            
        return Cd, Cl, Cm_0


class FlightController:
    """Controllore d'assetto e Allocatore di Controllo Riconfigurabile Dinamico (FTC)."""
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
        self.B_0 = np.array([25000.0, 25000.0, -30000.0, -30000.0])
        self.controller = FlightController()

    def equations_of_motion(self, t, state, t_fault, stuck_idx, delta_stuck):
        """Definisce il sistema di ODE a 7 stati per solve_ivp."""
        x, z, vx, vz, theta, q, e_int = state
        altitude = z
        
        rho = USStandardAtmosphere1976.get_density(altitude)
        
        # Raffica di vento 1-coseno
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
            
        theta_ref = get_theta_ref(t)
        error = theta_ref - theta
        
        # PID ottimizzato
        kp = 3.0e7
        kd = 2.0e7
        ki = 5.0e5
        
        e_int_max = np.radians(5.0)
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


# ==============================================================================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ==============================================================================
if __name__ == '__main__':
    print("==================================================================")
    print(" SIMULATORE DINAMICO 3-DOF STARSHIP - RIENTRO ATMOSFERICO E FTC")
    print("==================================================================")
    
    ship = Starship3DOF()
    
    # Condizioni Iniziali (x=0 m, z=15000 m, vx=120 m/s, vz=-70 m/s, theta=-20 deg)
    y0 = [0.0, 15000.0, 120.0, -70.0, np.radians(-20.0), 0.0, 0.0]
    t_span = (0.0, 40.0)
    t_eval = np.linspace(0.0, 40.0, 1000)
    
    # Scenario Guasto: t_fault = 10.0 s, Flap FL (indice 0) bloccato a +15.0 deg
    t_fault = 10.0
    stuck_idx = 0
    delta_stuck = np.radians(15.0)
    
    print("\n[1/3] Integrazione numerica delle equazioni del moto in corso...")
    sol = solve_ivp(
        ship.equations_of_motion, 
        t_span, 
        y0, 
        t_eval=t_eval,
        args=(t_fault, stuck_idx, delta_stuck),
        method='RK45'
    )
    
    print(f"      Simulazione completata con successo! ({len(sol.t)} passi temporali)")
    print(f"      - Quota iniziale: {y0[1]:.1f} m  -> Quota finale: {sol.y[1][-1]:.1f} m")
    print(f"      - Guasto iniettato: t = {t_fault:.1f} s (Flap FL bloccato a +15.0°)")
    print(f"      - Diagnosi FDI + Riconfigurazione: t = {t_fault + 0.5:.1f} s")
    
    print("\n[2/3] Generazione dei grafici di rientro...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Grafico 1: Quota
    ax1.plot(sol.t, sol.y[1] / 1000.0, 'b-', linewidth=2.0, label=r'Quota $z(t)$')
    ax1.axvline(t_fault, color='r', linestyle='--', alpha=0.7, label='Evento Guasto (t=10s)')
    ax1.axvline(t_fault + 0.5, color='g', linestyle=':', alpha=0.9, label='Riconfigurazione FTC (t=10.5s)')
    ax1.set_ylabel(r'Quota $z$ [km]')
    ax1.set_title("Traiettoria e Risposta al Guasto dell'Attuatore")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Grafico 2: Angolo di beccheggio vs Riferimento
    theta_deg = np.degrees(sol.y[4])
    theta_ref_deg = np.degrees([get_theta_ref(t_val) for t_val in sol.t])
    
    ax2.plot(sol.t, theta_deg, 'k-', linewidth=2.0, label=r'Beccheggio effettivo $\theta(t)$')
    ax2.plot(sol.t, theta_ref_deg, 'r--', linewidth=1.5, label=r'Riferimento $\theta_{\mathrm{ref}}$')
    ax2.axvline(t_fault, color='r', linestyle='--', alpha=0.7)
    ax2.axvline(t_fault + 0.5, color='g', linestyle=':', alpha=0.9)
    ax2.set_xlabel(r'Tempo $t$ [s]')
    ax2.set_ylabel(r'Assetto $\theta$ [deg]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.path.expanduser('~/Desktop')

    output_png = os.path.join(base_dir, 'simulation_results.png')
    
    plt.savefig(output_png, dpi=300)
    print(f"[3/3] Grafico salvato con successo in:\n      {output_png}")
    print("\nSimulazione conclusa. Mostro la figura...")
    plt.show()
