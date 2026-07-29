import numpy as np

class ControlAllocator:
    def __init__(self, d_min_deg=-40.0, d_max_deg=40.0):
        # Inizializza l'allocatore con i limiti fisici dei flap (rad)
        self.delta_min = np.radians(d_min_deg)
        self.delta_max = np.radians(d_max_deg)
        
    def allocate(self, M_cmd, B_current, fault_status=None):
        """
        M_cmd: Momento di beccheggio richiesto [N*m]
        B_current: Matrice di efficacia attuale (1x4) [N*m/rad]
        """
        # Caso 1: Funzionamento Nominale (4 attuatori attivi)
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
            
        # Applicazione dei vincoli di saturazione geometrica (Clamping)
        return np.clip(u_cmd, self.delta_min, self.delta_max)
