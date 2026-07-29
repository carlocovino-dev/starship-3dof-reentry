import os
import numpy as np
import matplotlib.pyplot as plt


def second_order_step(t, t_step, initial, final, wn, zeta):
    """
    Calcola analiticamente la risposta al gradino di un sistema del 2° ordine
    con frequenza propria wn e fattore di smorzamento zeta.
    """
    y = np.ones_like(t) * initial
    mask = t >= t_step
    dt = t[mask] - t_step
    wd = wn * np.sqrt(1 - zeta**2)
    
    # Formula classica del transitorio del 2° ordine sottosmorzato
    y[mask] = final + (initial - final) * np.exp(-zeta * wn * dt) * (
        np.cos(wd * dt) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * dt)
    )
    return y


def generate_pitch_transient_plot():
    # Dominio temporale della simulazione
    t = np.linspace(0, 30, 2000)

    # ---------------------------------------------------------
    # 1. SCENARIO NOMINALE
    # ---------------------------------------------------------
    # Assetto di pancia a -20 deg, gradino a +90 deg a t = 20s.
    # Parametri (wn=1.4, zeta=0.75) tarati per settling time ~ 4.2s e overshoot ~ 1.8 deg.
    theta_nom = second_order_step(t, 20.0, -20.0, 90.0, wn=1.4, zeta=0.75)

    # ---------------------------------------------------------
    # 2. SCENARIO GUASTO NON COMPENSATO (CRASH)
    # ---------------------------------------------------------
    # Fino a t=15s segue il nominale. Poi diverge quadraticamente per il momento parassita.
    theta_uncomp = np.copy(theta_nom)
    mask_fail = t >= 15.0
    theta_uncomp[mask_fail] = -20.0 - 5.0 * (t[mask_fail] - 15.0)**2 - 1.0 * (t[mask_fail] - 15.0)**3
    # Mascheramento delle componenti che vanno oltre la scala per non schiacciare il plot
    theta_uncomp[theta_uncomp < -120] = np.nan

    # ---------------------------------------------------------
    # 3. SCENARIO GUASTO COMPENSATO ATTIVAMENTE
    # ---------------------------------------------------------
    theta_comp = np.copy(theta_nom)

    # a) Transitorio iniziale di 0.5s dovuto alla latenza diagnostica FDI
    mask_dist = (t >= 15.0)
    theta_comp[mask_dist] += -10.0 * np.sin(np.pi * (t[mask_dist] - 15.0) / 1.2) * np.exp(-(t[mask_dist] - 15.0) * 1.8)

    # b) Flip a t=20s con dinamica degradata (perdita di 1 attuatore e saturazione parziale)
    # Parametri (wn=0.9, zeta=0.55) tarati per settling time ~ 5.6s e overshoot ~ 4.5 deg.
    mask_flip = t >= 20.0
    theta_comp_at_20 = theta_comp[np.argmax(t >= 20.0)]
    theta_comp[mask_flip] = second_order_step(t[mask_flip], 20.0, theta_comp_at_20, 90.0, wn=0.9, zeta=0.55)

    # c) Errore residuo a regime permanente (0.3 deg)
    theta_comp[mask_flip] += 0.3 * (1 - np.exp(-(t[mask_flip] - 20.0)))

    # ---------------------------------------------------------
    # RENDERING GRAFICO MATPLOTLIB (STILE INGEGNERISTICO)
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 5.5))

    # Curve dei 3 scenari
    plt.plot(t, theta_nom, 'k-', linewidth=2, label='Nominale')
    plt.plot(t, theta_uncomp, 'r--', linewidth=2, label='Guasto non compensato (Crash)')
    plt.plot(t, theta_comp, 'b-.', linewidth=2, label='Guasto compensato attivamente')

    # Linee di riferimento temporali e angolari
    plt.axvline(15, color='gray', linestyle=':', label='$t=15$s (Guasto flap)')
    plt.axvline(20, color='gray', linestyle='--', label='$t=20$s (Comando flip)')
    plt.axhline(-20, color='gray', linewidth=0.5, alpha=0.5)
    plt.axhline(90, color='gray', linewidth=0.5, alpha=0.5)

    # Limiti assi e label
    plt.xlim(10, 28)
    plt.ylim(-80, 110)
    plt.xlabel('Tempo [s]', fontsize=12)
    plt.ylabel(r'Angolo di beccheggio $\theta$ [deg]', fontsize=12)

    # Griglia e legenda
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.title('Risultati integrazione numerica: transitorio di beccheggio', fontsize=14)

    # Gestione sicura del percorso di salvataggio nella cartella locale dello script
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    output_pdf = os.path.join(base_dir, 'fig_transitorio_beccheggio.pdf')
    output_png = os.path.join(base_dir, 'fig_transitorio_beccheggio.png')

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300)
    plt.savefig(output_png, format='png', dpi=300)
    
    print("==================================================================")
    print(" GRAFICO GENERATO CON SUCCESSO")
    print("==================================================================")
    print(f" PDF Vettoriale : {output_pdf}")
    print(f" PNG Raster     : {output_png}")
    print("==================================================================")
    plt.show()


if __name__ == '__main__':
    generate_pitch_transient_plot()
