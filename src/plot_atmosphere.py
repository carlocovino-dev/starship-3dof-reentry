import os
import matplotlib.pyplot as plt
import numpy as np

class Atmosphere1976:
    """Modello US Standard Atmosphere 1976 completo fino a 80 km.
    
    Progettato per query puntuali ed essere importato come modulo nel simulatore 3-DOF.
    """
    R = 287.05      # Costante specifica dell'aria secca [J/(kg*K)]
    g = 9.80665     # Accelerazione di gravita standard [m/s^2]
    gamma = 1.4     # Rapporto dei calori specifici

    @staticmethod
    def get_properties(alt):
        """Restituisce [rho (kg/m^3), p (kPa), a (m/s), T (K)] per un'altitudine z [m]."""
        z = max(0.0, alt)
        if z < 11000.0:  # Troposfera
            T = 288.15 - 0.0065 * z
            rho = 1.225 * (T / 288.15) ** ((Atmosphere1976.g / (0.0065 * Atmosphere1976.R)) - 1.0)
            p = rho * Atmosphere1976.R * T
        elif z < 20000.0:  # Bassa Stratosfera (Isoterma)
            T = 216.65
            rho_11 = 0.36391
            rho = rho_11 * np.exp(-Atmosphere1976.g * (z - 11000.0) / (Atmosphere1976.R * T))
            p = rho * Atmosphere1976.R * T
        else:  # Alta Stratosfera / Mesosfera inferiore
            T = 216.65 + 0.001 * (z - 20000.0)
            rho = 0.08803 * (T / 216.65) ** (-(Atmosphere1976.g / (0.001 * Atmosphere1976.R)) - 1.0)
            p = rho * Atmosphere1976.R * T
            
        a = np.sqrt(Atmosphere1976.gamma * Atmosphere1976.R * T)
        return rho, p / 1000.0, a, T  # p in kPa


def plot_atmosphere_profiles(output_dir=None):
    """Genera e salva la figura pubblicabile a 4 pannelli."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Calcolo dati
    altitudes_km = np.linspace(0, 80, 1000)
    altitudes_m = altitudes_km * 1000.0

    densities, pressures, speeds_of_sound, temperatures = [], [], [], []

    for z in altitudes_m:
        rho, p_kPa, a, T = Atmosphere1976.get_properties(z)
        densities.append(rho)
        pressures.append(p_kPa)
        speeds_of_sound.append(a)
        temperatures.append(T)

    # 2. Configurazione stile accademico avanzato
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'mathtext.fontset': 'cm'
    })

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.8), sharey=True, layout='constrained')
    ax1, ax2, ax3, ax4 = axes

    # Definizioni sfondi per strati atmosferici
    layers = [
        (0, 11, 'Troposfera', '#EBF5FB'),
        (11, 20, 'Bassa Stratosfera', '#F4ECF7'),
        (20, 50, 'Stratosfera', '#FEF9E7'),
        (50, 80, 'Mesosfera', '#E8F8F5')
    ]

    for ax in axes:
        for z_bot, z_top, name, bg_color in layers:
            ax.axhspan(z_bot, z_top, color=bg_color, alpha=0.75, zorder=0)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#B0BEC5', alpha=0.7)
        ax.tick_params(direction='in', top=True, right=True)

    # --- PANEL 1: Temperatura ---
    ax1.plot(temperatures, altitudes_km, color='#B03A2E', linewidth=2.2, zorder=3)
    ax1.set_xlabel('Temperatura $T$ [K]')
    ax1.set_ylabel('Altitudine geometrica $z$ [km]')
    ax1.set_title('(a) Profilo di Temperatura', fontweight='bold', pad=8)
    ax1.set_xlim(195, 305)
    ax1.plot(288.15, 0, 'o', color='#7B241C', markersize=4.5, zorder=4)
    ax1.text(285, 2.5, r'$T_0 = 288.15$ K', fontsize=7.5, color='#7B241C', ha='right', fontweight='bold')

    # --- PANEL 2: Densità ---
    ax2.semilogx(densities, altitudes_km, color='#1F618D', linewidth=2.2, zorder=3)
    ax2.set_xlabel(r'Densità $\rho$ [kg/m$^3$]')
    ax2.set_title(r'(b) Profilo di Densità', fontweight='bold', pad=8)
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#CFD8DC', alpha=0.6)
    ax2.plot(1.225, 0, 'o', color='#154360', markersize=4.5, zorder=4)
    ax2.text(0.8, 2.5, r'$\rho_0 = 1.225$', fontsize=7.5, color='#154360', ha='right', fontweight='bold')

    # --- PANEL 3: Pressione ---
    ax3.semilogx(pressures, altitudes_km, color='#6C3483', linewidth=2.2, zorder=3)
    ax3.set_xlabel('Pressione $p$ [kPa]')
    ax3.set_title('(c) Profilo di Pressione', fontweight='bold', pad=8)
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#CFD8DC', alpha=0.6)
    ax3.plot(101.325, 0, 'o', color='#512E5F', markersize=4.5, zorder=4)
    ax3.text(60, 2.5, r'$p_0 = 101.3$ kPa', fontsize=7.5, color='#512E5F', ha='right', fontweight='bold')

    # --- PANEL 4: Velocità del suono ---
    ax4.plot(speeds_of_sound, altitudes_km, color='#1E8449', linewidth=2.2, zorder=3)
    ax4.set_xlabel('Velocità del suono $a$ [m/s]')
    ax4.set_title('(d) Velocità del Suono', fontweight='bold', pad=8)
    ax4.set_xlim(280, 350)
    ax4.plot(340.29, 0, 'o', color='#114B27', markersize=4.5, zorder=4)
    ax4.text(338, 2.5, r'$a_0 = 340.3$ m/s', fontsize=7.5, color='#114B27', ha='right', fontweight='bold')

    # Etichette strati
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B0BEC5', alpha=0.85, linewidth=0.6)
    ax4.text(284, 5.5, 'Troposfera', fontsize=8, fontstyle='italic', color='#2C3E50', bbox=bbox_props)
    ax4.text(284, 15.5, 'Bassa Stratosfera', fontsize=8, fontstyle='italic', color='#2C3E50', bbox=bbox_props)
    ax4.text(284, 35.0, 'Stratosfera', fontsize=8, fontstyle='italic', color='#2C3E50', bbox=bbox_props)
    ax4.text(284, 65.0, 'Mesosfera', fontsize=8, fontstyle='italic', color='#2C3E50', bbox=bbox_props)

    for ax in axes:
        for z_bound in [11, 20, 50]:
            ax.axhline(z_bound, color='#546E7A', linestyle='--', linewidth=0.8, zorder=2)

    pdf_path = os.path.join(output_dir, 'us_standard_atmosphere_1976.pdf')
    jpg_path = os.path.join(output_dir, 'us_standard_atmosphere_1976.jpg')

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(jpg_path, format='jpg', bbox_inches='tight', dpi=300)
    
    print(f"Grafici salvati con successo in:\n - {pdf_path}\n - {jpg_path}")


if __name__ == '__main__':
    plot_atmosphere_profiles()
