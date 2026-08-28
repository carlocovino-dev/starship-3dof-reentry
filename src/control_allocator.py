import numpy as np


class ControlAllocator:
    def __init__(self, d_min_deg=-40.0, d_max_deg=40.0):
        # Inizializza l'allocatore con i limiti fisici dei flap (espressi in rad)
        self.delta_min = np.radians(d_min_deg)
        self.delta_max = np.radians(d_max_deg)
        
    def allocate(self, M_cmd, B_current, fault_status=None):
        """
        M_cmd: Momento di beccheggio richiesto [N*m]
        B_current: Matrice di efficacia attuale (1x4) [N*m/rad]
        """
        # Caso 1: Funzionamento nominale (4 attuatori attivi)
        if fault_status is None:
            # Matrice dei pesi (penalizza flap anteriori per limitare il drag)
            W = np.diag([1.5, 1.5, 1.0, 1.0]) 
            W_inv = np.linalg.inv(W)
            
            # Pseudo-inversa pesata: u = W^-1 * B^T * (B * W^-1 * B^T)^-1 * M_cmd
            temp = B_current.dot(W_inv).dot(B_current.T)
            u_cmd = W_inv.dot(B_current.T) * (M_cmd / temp)
            
        # Caso 2: Riconfigurazione con guasto (Single-flap failure)
        else:
            stuck_idx = fault_status['index'] # Indice flap guasto (0 a 3)
            delta_stuck = np.radians(fault_status['angle_deg'])
            
            # Riduzione della matrice B
            B_act = np.delete(B_current, stuck_idx)
            M_reconfig = M_cmd - B_current[stuck_idx] * delta_stuck
            
            # Riduzione della matrice dei pesi
            W_nom = np.diag([1.5, 1.5, 1.0, 1.0])
            W_act = np.delete(np.delete(W_nom, stuck_idx, axis=0), stuck_idx, axis=1)
            W_act_inv = np.linalg.inv(W_act)
            
            # Ricalcolo sui 3 attuatori operativi
            temp = B_act.dot(W_act_inv).dot(B_act.T)
            u_act = W_act_inv.dot(B_act.T) * (M_reconfig / temp)
            
            # Ricostruzione del vettore completo a 4 dimensioni
            u_cmd = np.insert(u_act, stuck_idx, delta_stuck)
            
        # Applicazione dei vincoli di saturazione geometrica
        return np.clip(u_cmd, self.delta_min, self.delta_max)


# ==============================================================================
# BLOCCO DI TEST PER L'ESECUZIONE
# ==============================================================================
if __name__ == '__main__':
    print("==================================================================")
    print(" TEST CONTROL ALLOCATOR (Pseudo-inversa pesata & FTC)")
    print("==================================================================")

    allocator = ControlAllocator(d_min_deg=-40.0, d_max_deg=40.0)

    # Parametri di prova (Pressione dinamica q_inf = 3000 Pa)
    q_inf = 3000.0 
    B_0 = np.array([25000.0, 25000.0, -30000.0, -30000.0]) # N*m/Pa/rad
    B_current = B_0 * q_inf

    # Richiesta momento di beccheggio: +5,000,000 N*m
    M_cmd = 5.0e6 

    # 1. TEST CASO NOMINALE
    u_nominal = allocator.allocate(M_cmd, B_current, fault_status=None)
    u_nom_deg = np.degrees(u_nominal)

    print("\n--- 1. CASO NOMINALE (4 Flap Attivi) ---")
    print(f"Momento richiesto M_cmd: {M_cmd:.2e} N*m")
    print(f"Deflessioni allocate [deg]:")
    print(f"  Flap 0 (Front Left)  : {u_nom_deg[0]:+6.2f}°")
    print(f"  Flap 1 (Front Right) : {u_nom_deg[1]:+6.2f}°")
    print(f"  Flap 2 (Aft Left)    : {u_nom_deg[2]:+6.2f}°")
    print(f"  Flap 3 (Aft Right)   : {u_nom_deg[3]:+6.2f}°")

    # 2. TEST CASO GUASTO (FTC)
    fault_scenario = {'index': 0, 'angle_deg': 25.0} # Flap 0 bloccato a +25 gradi
    u_fault = allocator.allocate(M_cmd, B_current, fault_status=fault_scenario)
    u_fault_deg = np.degrees(u_fault)

    print("\n--- 2. CASO GUASTO E RICONFIGURAZIONE FTC ---")
    print(f"Guasto iniettato: Flap {fault_scenario['index']} bloccato a +{fault_scenario['angle_deg']}°")
    print(f"Deflessioni riconfigurate [deg]:")
    print(f"  Flap 0 (Guasto)      : {u_fault_deg[0]:+6.2f}°")
    print(f"  Flap 1 (Operativo)   : {u_fault_deg[1]:+6.2f}°")
    print(f"  Flap 2 (Operativo)   : {u_fault_deg[2]:+6.2f}°")
    print(f"  Flap 3 (Operativo)   : {u_fault_deg[3]:+6.2f}°")
