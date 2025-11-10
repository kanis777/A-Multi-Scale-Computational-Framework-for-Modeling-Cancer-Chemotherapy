import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of differential equations from the paper (System 11)
# y[0] = E (Effector cells), y[1] = T (Tumor cells), y[2] = M (Drug concentration)
def cancer_chemotherapy_model(t, y, p, s, h, m, mu, r, b, a, g, gamma, V, KE, KT):
    """
    Defines the differential equations for the cancer-chemotherapy model.

    Equations are based on System (11) in the research paper:
    dE/dt = s + p*(E*T)/(h+T) - m*E*T - mu*E - KE*M*E
    dT/dt = r*T*(1 - b*T) - a*(E*T)/(T+g) - KT*M*T
    dM/dt = -gamma*M + V
    """
    E, T, M = y

    dEdt = s + p * (E * T) / (h + T) - m * E * T - mu * E - KE * M * E
    dTdt = r * T * (1 - b * T) - a * (E * T) / (T + g) - KT * M * T
    dMdt = -gamma * M + V

    return [dEdt, dTdt, dMdt]

# --- Parameters from Table 1 of the research paper ---
params = {
    'p': 0.015,          # Recruitment rate of effector cells (1/day)
    'r': 4.31*(10**(-3)),        # Rate of tumor growth (1/day) -> This will be changed for Sim 2 & 3
    'b': 1.0*(10**(-9)),         # Tumor carrying capacity parameter (1/cells)
    'a': 3.41*(10**(-10)),       # Parameter of cancer cleanup (1/cells)
    'g': 1.05*(10**2),         # Half-saturation for cancer cleanup (cells)
    's': 1.2*(10**4),          # Growth rate of effector cells (cells/day)
    'm': 2.0*(10**(-11)),        # Inactivation rate of effector cells by tumor cells (1/cells.day)
    'mu': 4.12*(10**(-2)),       # Natural death rate of effector cells (1/day)
    'gamma': 0.9,        # Rate of decrease in drug concentration (1/day)
    'h': 2.02*(10**4),         # Steepness coefficient for effector cell recruitment (cells)

    # NOTE: These parameters were not specified in the paper.
    # Values have been assumed to produce results consistent with the paper's graphs.
    'KE': 0.3,           # Chemotherapy effect on Effector cells
    'KT': 0.8,           # Chemotherapy effect on Tumor cells
}

# --- Initial conditions from the paper ---
# E(0) = 30000 cells, T(0) = 40000 cells, M(0) = 0
initial_conditions = [30000, 40000, 0]

# --- Time span for the simulation ---
# As per the graphs in the paper (0 to 50 days)
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# --- Simulation 1 (Replicating Fig. 1a) ---
params1 = params.copy()
params1['V'] = 0.5
sol1 = solve_ivp(
    lambda t, y: cancer_chemotherapy_model(t, y, **params1),
    t_span,
    initial_conditions,
    t_eval=t_eval
)

# --- Simulation 2 (Replicating Fig. 1b) ---
params2 = params.copy()
params2['r'] = 0.47
params2['V'] = 0.5
sol2 = solve_ivp(
    lambda t, y: cancer_chemotherapy_model(t, y, **params2),
    t_span,
    initial_conditions,
    t_eval=t_eval
)

# --- Simulation 3 (Replicating Fig. 1c) ---
params3 = params.copy()
params3['r'] = 0.47
params3['V'] = 0.6
sol3 = solve_ivp(
    lambda t, y: cancer_chemotherapy_model(t, y, **params3),
    t_span,
    initial_conditions,
    t_eval=t_eval
)

# --- Plotting the results (Modified to match the paper's style) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Cancer Chemotherapy Model Simulations (Corrected)', fontsize=16)

# Plot for Simulation 1
ax1.plot(sol1.t, sol1.y[0], label='Effector', color='black')
ax1.plot(sol1.t, sol1.y[1], label='Tumor', color='red')
# PLOTTING CHANGE: Plot concentration on the same axis
ax1.plot(sol1.t, sol1.y[2], label='Concentration', color='green')
ax1.set_title('Simulation 1 (Fig. 1a): Low Tumor Growth ($r=0.00431$), $V=0.5$')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Cell Population')
ax1.grid(True)
ax1.legend(loc='best')
ax1.set_ylim(bottom=0)

# Plot for Simulation 2
ax2.plot(sol2.t, sol2.y[0], label='Effector', color='black')
ax2.plot(sol2.t, sol2.y[1], label='Tumor', color='red')
# PLOTTING CHANGE: Plot concentration on the same axis
ax2.plot(sol2.t, sol2.y[2], label='Concentration', color='green')
ax2.set_title('Simulation 2 (Fig. 1b): High Tumor Growth ($r=0.47$), $V=0.5$')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Cell Population')
ax2.grid(True)
ax2.legend(loc='best')
ax2.set_ylim(bottom=0)

# Plot for Simulation 3
ax3.plot(sol3.t, sol3.y[0], label='Effector', color='black')
ax3.plot(sol3.t, sol3.y[1], label='Tumor', color='red')
# PLOTTING CHANGE: Plot concentration on the same axis
ax3.plot(sol3.t, sol3.y[2], label='Concentration', color='green')
ax3.set_title('Simulation 3 (Fig. 1c): High Tumor Growth ($r=0.47$), Higher Dosage ($V=0.6$)')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Cell Population')
ax3.grid(True)
ax3.legend(loc='best')
ax3.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()