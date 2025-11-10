import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Model Parameters ---
# Parameters are based on Table 1 in the research paper.
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

    'KE': 0.3,           # Chemotherapy effect on Effector cells
    'KT': 0.8,           # Chemotherapy effect on Tumor cells
}

# --- 2. Define SDE Drift and Diffusion Functions ---

def drift(y, t, p, V_M_func):
    """ The deterministic part of the SDE (the original ODEs). """
    E, T, M = np.maximum(0, y) # Ensure no negative values
    current_V_M = V_M_func(t)
    dE_dt = p['s'] - p['mu'] * E + (p['p'] * E * T) / (p['h'] + T) - p['m'] * E * T - p['KE'] * E * M
    dT_dt = p['r'] * T * (1 - p['b'] * T) - (p['a'] * E * T) / (T + p['g']) - p['KT'] * T * M
    dM_dt = -p['gamma'] * M + current_V_M
    return np.array([dE_dt, dT_dt, dM_dt])

def diffusion(y, t, p, V_M_func):
    """ The stochastic part of the SDE (noise terms). """
    E, T, M = np.maximum(0, y) # Ensure no negative values
    current_V_M = V_M_func(t)
    var_E = p['s'] + p['mu'] * E + (p['p'] * E * T) / (p['h'] + T) + p['m'] * E * T + p['KE'] * E * M
    var_T = abs(p['r'] * T * (1 - p['b'] * T)) + (p['a'] * E * T) / (T + p['g']) + p['KT'] * T * M
    var_M = p['gamma'] * M + current_V_M
    return np.diag([np.sqrt(var_E), np.sqrt(var_T), np.sqrt(var_M)])

# --- 3. SDE Numerical Solver (Euler-Maruyama) ---

def sde_solver(y0, t_points, drift_func, diffusion_func, params, V_M_func):
    """ Solves a system of SDEs using the Euler-Maruyama method. """
    num_vars = len(y0)
    num_steps = len(t_points)
    y_out = np.zeros((num_steps, num_vars))
    y_out[0] = y0
    for i in range(1, num_steps):
        t = t_points[i-1]
        dt = t_points[i] - t
        dW = np.random.normal(0.0, np.sqrt(dt), num_vars)
        drift_term = drift_func(y_out[i-1], t, params, V_M_func)
        diffusion_term = diffusion_func(y_out[i-1], t, params, V_M_func) @ dW
        y_out[i] = y_out[i-1] + drift_term * dt + diffusion_term
        y_out[i] = np.maximum(0, y_out[i]) # Ensure no negative populations
    return y_out

# --- 4. Run All Three Simulations ---

# Common simulation settings
t = np.linspace(0, 50, 500)
initial_conditions = [30000, 40000, 0]
num_simulations = 50 # Number of paths for the ensemble

# --- Scenario 1: Low tumor growth (r=0.00431), V=0.5 ---
params_sim1 = params.copy()
V_M_sim1_func = lambda time: 0.5
ensemble1 = np.array([sde_solver(initial_conditions, t, drift, diffusion, params_sim1, V_M_sim1_func) for _ in range(num_simulations)])
mean1, std1 = np.mean(ensemble1, axis=0), np.std(ensemble1, axis=0)

# [cite_start]--- Scenario 2: High tumor growth (r=0.47), V=0.5 --- [cite: 229]
params_sim2 = params.copy()
params_sim2['r'] = 0.47
V_M_sim2_func = lambda time: 0.5
ensemble2 = np.array([sde_solver(initial_conditions, t, drift, diffusion, params_sim2, V_M_sim2_func) for _ in range(num_simulations)])
mean2, std2 = np.mean(ensemble2, axis=0), np.std(ensemble2, axis=0)

# [cite_start]--- Scenario 3: High tumor growth (r=0.47), V=0.6 --- [cite: 235]
params_sim3 = params.copy()
params_sim3['r'] = 0.47
V_M_sim3_func = lambda time: 0.6
ensemble3 = np.array([sde_solver(initial_conditions, t, drift, diffusion, params_sim3, V_M_sim3_func) for _ in range(num_simulations)])
mean3, std3 = np.mean(ensemble3, axis=0), np.std(ensemble3, axis=0)

# --- 5. Plotting the Results ---

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
fig.suptitle('SDE Simulations of Cancer Chemotherapy Model', fontsize=16)

# Plot for Scenario 1
ax1.set_title('Scenario 1 (Fig. 1a): Low Tumor Growth ($r=0.00431$), $V=0.5$')
ax1.plot(t, mean1[:, 0], color='black', label='Mean Effector (E)')
ax1.fill_between(t, mean1[:, 0] - std1[:, 0], mean1[:, 0] + std1[:, 0], color='gray', alpha=0.3)
ax1.plot(t, mean1[:, 1], color='red', label='Mean Tumor (T)')
ax1.fill_between(t, mean1[:, 1] - std1[:, 1], mean1[:, 1] + std1[:, 1], color='pink', alpha=0.3, label='Std. Dev.')
ax1.set_ylabel('Cell Population')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(bottom=0)

# Plot for Scenario 2
ax2.set_title('Scenario 2 (Fig. 1b): High Tumor Growth ($r=0.47$), $V=0.5$')
ax2.plot(t, mean2[:, 0], color='black', label='Mean Effector (E)')
ax2.fill_between(t, mean2[:, 0] - std2[:, 0], mean2[:, 0] + std2[:, 0], color='gray', alpha=0.3)
ax2.plot(t, mean2[:, 1], color='red', label='Mean Tumor (T)')
ax2.fill_between(t, mean2[:, 1] - std2[:, 1], mean2[:, 1] + std2[:, 1], color='pink', alpha=0.3, label='Std. Dev.')
ax2.set_ylabel('Cell Population')
ax2.legend()
ax2.grid(True)
ax2.set_ylim(bottom=0)

# Plot for Scenario 3
ax3.set_title('Scenario 3 (Fig. 1c): High Tumor Growth ($r=0.47$), Higher Dosage ($V=0.6$)')
ax3.plot(t, mean3[:, 0], color='black', label='Mean Effector (E)')
ax3.fill_between(t, mean3[:, 0] - std3[:, 0], mean3[:, 0] + std3[:, 0], color='gray', alpha=0.3)
ax3.plot(t, mean3[:, 1], color='red', label='Mean Tumor (T)')
ax3.fill_between(t, mean3[:, 1] - std3[:, 1], mean3[:, 1] + std3[:, 1], color='pink', alpha=0.3, label='Std. Dev.')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Cell Population')
ax3.legend()
ax3.grid(True)
ax3.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()