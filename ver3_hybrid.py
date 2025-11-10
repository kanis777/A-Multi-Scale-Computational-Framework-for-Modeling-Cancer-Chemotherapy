import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import random

# ==============================================================================
# PARAMETERS AND MODEL 1: 'orig' (ODE)
# ==============================================================================

def get_base_parameters():
    """Returns the baseline parameter dictionary based on Lestari et al."""
    return {
        's': 1.2e4,       # growth rate of effector cells (cells/day)
        'p': 0.015,       # degree of recruitment of E-cells (1/day)
        'h': 2.02e4,      # steepness coeff. of recruitment (cells)
        'm': 2e-11,       # degree of inactivation of E-cells (1/cells.day)
        'mu': 4.12e-2,    # rate of natural demise of E-cells (1/day)
        'r': 4.31e-3,     # rate of tumor growth (1/day)
        'b': 1e-9,        # capacity of tumor cell (1/cells)
        'a': 3.41e-10,    # parameter of cancer cleanup (1/cells)
        'g': 1e5,         # half-saturation for cleanup (cells)
        'gamma': 0.9,     # rate of decrease in chemo (1/day)
        'KE': 0.3,        # Chemo effect on E-cells (assumed)
        'KT': 0.8,        # Chemo effect on T-cells (assumed)
        'V': 0.5,         # Chemo admin rate (Scenario 1)
        'K_cap': 1e9      # Carrying capacity K = 1/b
    }

def ode_model(t, y, p):
    """
    Defines the deterministic ODE system (Lestari et al., System 11).
    Args:
        t (float): Time
        y (list): State vector
        p (dict): Parameter dictionary
    """
    E, T, M = y
    E, T, M = max(0, E), max(0, T), max(0, M)

    dEdt = p['s'] + (p['p'] * E * T) / (p['h'] + T) - p['m'] * E * T - p['mu'] * E - p['KE'] * M * E
    # ---  Used p['KT'] instead of p ---
    dTdt = p['r'] * T * (1 - p['b'] * T) - (p['a'] * E * T) / (p['g'] + T) - p['KT'] * M * T
    dMdt = -p['gamma'] * M + p['V']

    return [dEdt, dTdt, dMdt]

def run_ode_simulation(params, y0, t_span, t_eval):
    """Runs the ODE simulation using solve_ivp."""
    sol = solve_ivp(
        fun=ode_model, t_span=t_span, y0=y0, t_eval=t_eval,
        args=(params,), method='RK45'
    )
    return sol.t, sol.y

# ==============================================================================
# MODEL 2: 'sde' (SDE)
# ==============================================================================

def sde_drift(y, t, p, T_surface=None):
    """
    Defines the drift component (deterministic part) of the SDE.
    Can be adapted for the hybrid model using T_surface.
    """
    E, T, M = y
    E, T, M = max(0, E), max(0, T), max(0, M)

    T_interaction = T_surface if T_surface is not None else T

    # E-cell rates
    s_rate = p['s']
    p_rate = (p['p'] * E * T_interaction) / (p['h'] + T_interaction)
    m_rate = -p['m'] * E * T_interaction
    mu_rate = -p['mu'] * E
    KE_rate = -p['KE'] * M * E

    # T-cell rates
    r_rate = p['r'] * T * (1 - p['b'] * T)
    a_rate = -(p['a'] * E * T) / (p['g'] + T)
    KT_rate = -p['KT'] * M * T

    # M-cell rates
    gamma_rate = -p['gamma'] * M
    V_rate = p['V']

    dEdt = s_rate + p_rate + m_rate + mu_rate + KE_rate
    dTdt = r_rate + a_rate + KT_rate  # Note: This is unused in hybrid model
    dMdt = gamma_rate + V_rate

    return np.array([dEdt, dTdt, dMdt])

def sde_diffusion(y, t, p, T_surface=None):
    """
    Defines the diffusion component (stochastic part) of the SDE.
    Uses the heuristic approximation (sum of absolute rates).
    """
    E, T, M = y
    E, T, M = max(0, E), max(0, T), max(0, M)

    T_interaction = T_surface if T_surface is not None else T

    # E-cell rates
    var_E = (
        abs(p['s']) +
        abs((p['p'] * E * T_interaction) / (p['h'] + T_interaction)) +
        abs(-p['m'] * E * T_interaction) +
        abs(-p['mu'] * E) +
        abs(-p['KE'] * M * E)
    )

    # T-cell rates
    var_T = (
        abs(p['r'] * T * (1 - p['b'] * T)) +
        abs(-(p['a'] * E * T) / (p['g'] + T)) +
        abs(-p['KT'] * M * T)
    )

    # M-cell rates
    var_M = abs(-p['gamma'] * M) + abs(p['V'])

    # --- Return sqrt of variance for SDE ---
    # Clamp to avoid sqrt(0) issues if rates are zero
    var_E = max(1e-10, var_E)
    var_T = max(1e-10, var_T)
    var_M = max(1e-10, var_M)

    return np.diag([np.sqrt(var_E), np.sqrt(var_T), np.sqrt(var_M)])

def sde_solver(y0, t_span, dt, drift_func, diffusion_func, params):
    """
    Solves the SDE system using the Euler-Maruyama method.
    """
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt, dt)
    num_vars = len(y0)
    y_out = np.zeros((len(t_eval), num_vars))
    y_out[0, :] = y0

    for i in range(1, len(t_eval)):
        t = t_eval[i-1]
        y_prev = y_out[i-1, :]

        dW = np.random.normal(0.0, np.sqrt(dt), num_vars)
        drift_term = drift_func(y_prev, t, params) * dt
        diffusion_term = diffusion_func(y_prev, t, params) @ dW

        y_next = y_prev + drift_term + diffusion_term
        y_out[i, :] = np.maximum(0, y_next)

    return t_eval, y_out

def run_sde_simulation(params, y0, t_span, dt, num_runs=50):
    """Runs an ensemble of SDE simulations."""
    results = []
    for _ in range(num_runs):
        t, y = sde_solver(y0, t_span, dt, sde_drift, sde_diffusion, params)
        results.append(y)
    return t, np.array(results)

# ==============================================================================
# MODEL 3: 'network' (Hybrid SDE + Agent-Based Network)
# ==============================================================================

def generate_tumor_network(n_nodes, radius):
    """
    Generates a spatial graph for the tumor.
    Nodes have 'pos', 'is_alive', and 'path_len'.
    """
    G = nx.Graph()
    if n_nodes == 0:
        return G

    for i in range(n_nodes):
        G.add_node(i, pos=np.random.rand(2), is_alive=True)

    pos = nx.get_node_attributes(G, 'pos')
    for i in G.nodes():
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < radius:
                G.add_edge(i, j)

    # Find source node (closest to origin) and compute path lengths
    source_node = min(G.nodes(), key=lambda n: np.linalg.norm(pos[n]))
    nx.set_node_attributes(G, {source_node: True}, name='is_source')

    path_lengths = nx.shortest_path_length(G, source=source_node)

    # Handle disconnected nodes
    for node in G.nodes():
        if node not in path_lengths:
            path_lengths[node] = 999 # Assign a large path length

    nx.set_node_attributes(G, path_lengths, name='path_len')
    return G

class HybridSimulator:
    """
    Manages the state and stepping of the hybrid SDE-network model.
    """
    def __init__(self, y0_globals, G, params, dt):
        self.E = y0_globals[0]
        self.M = y0_globals[1]
        self.G = G
        self.params = params
        self.dt = dt
        self.history = []
        self.current_time = 0

        self.E_idx, self.M_idx = 0, 1 # Indices for a 2-var system

    def get_living_nodes(self):
        """Returns a list of nodes where is_alive is True."""
        return [n for n, data in self.G.nodes(data=True) if data['is_alive']]

    def get_surface_nodes(self, living_nodes):
        """Identifies surface nodes (degree <= average degree)."""
        if not living_nodes:
            return [], {}

        living_subgraph = self.G.subgraph(living_nodes)
        if living_subgraph.number_of_nodes() == 0:
            return [], {}

        degrees = dict(living_subgraph.degree())
        if not degrees:
            return [], {}

        avg_degree = np.mean(list(degrees.values()))

        surface_nodes = [n for n, deg in degrees.items() if deg <= avg_degree]
        return surface_nodes, degrees

    def step_sde_globals(self, T_surface_count):
        """
        Advances E and M one step using Euler-Maruyama.
        Feeds T_surface_count into the drift/diffusion functions.
        """
        y_prev = np.array([self.E, self.M])
        p_ = self.params
        t = self.current_time

        dummy_y = np.array([self.E, 0, self.M]) # E, T_placeholder, M

        full_drift = sde_drift(dummy_y, t, p_, T_surface=T_surface_count)
        full_diff_matrix = sde_diffusion(dummy_y, t, p_, T_surface=T_surface_count)

        # Extract only E (index 0) and M (index 2) components
        drift_vec = np.array([full_drift[0], full_drift[2]])

        # --- Correctly access diagonal elements ---
        diff_vec = np.array([full_diff_matrix[0, 0], full_diff_matrix[2, 2]])
        diff_matrix = np.diag(diff_vec)

        # Euler-Maruyama step for E and M
        dW = np.random.normal(0.0, np.sqrt(self.dt), 2)
        drift_term = drift_vec * self.dt
        diffusion_term = diff_matrix @ dW

        y_next = y_prev + drift_term + diffusion_term
        self.E, self.M = np.maximum(0, y_next)

    def step_tumor_proliferation(self, living_nodes, K_cap):
        """
        Implements agent-based tumor growth using r and b.
        """
        new_nodes_info = []
        N_alive = len(living_nodes)
        p_ = self.params

        prob_divide = p_['r'] * (1 - N_alive / K_cap) * self.dt

        if prob_divide <= 0:
            return

        for node_id in living_nodes:
            if random.random() < prob_divide:
                new_node_id = self.G.number_of_nodes()
                parent_pos = self.G.nodes[node_id]['pos']
                parent_path_len = self.G.nodes[node_id]['path_len']

                new_pos = parent_pos + np.random.randn(2) * 0.01 # Small nudge

                new_nodes_info.append({
                    'id': new_node_id,
                    'pos': new_pos,
                    'parent': node_id,
                    'path_len': parent_path_len + 1 # Simple heuristic
                })

        # --- Add new nodes to the graph ---
        pos_attr = nx.get_node_attributes(self.G, 'pos')
        for info in new_nodes_info:
            self.G.add_node(info['id'], pos=info['pos'], is_alive=True, path_len=info['path_len'])
            pos_attr[info['id']] = info['pos']

            self.G.add_edge(info['id'], info['parent'])

            for neighbor in self.G.neighbors(info['parent']):
                if neighbor not in pos_attr: continue
                dist = np.linalg.norm(info['pos'] - pos_attr[neighbor])
                if 'network_radius' in self.params and dist < self.params['network_radius']:
                    self.G.add_edge(info['id'], neighbor)

    def step_tumor_killing(self, living_nodes, surface_nodes):
        """
        Applies probabilistic killing from Immune (surface)
        and Chemo (gradient) attacks.
        """
        p_ = self.params
        T_surface_count = len(surface_nodes)

        P_kill_E = 0
        if T_surface_count > 0:
            P_kill_E = (p_['a'] * self.E / (p_['g'] + T_surface_count)) * self.dt

        nodes_to_kill = []
        for node_id in living_nodes:

            # 2a. Immune Killing (Surface only)
            if node_id in surface_nodes:
                if random.random() < P_kill_E:
                    nodes_to_kill.append(node_id)
                    continue

            # 2b. Chemo Killing (All nodes, gradient-dependent)
            path_len = self.G.nodes[node_id]['path_len']
            alpha = 0.1 # Chemo decay factor
            M_local = self.M * np.exp(-alpha * path_len)

            P_kill_M = (p_['KT'] * M_local) * self.dt

            if random.random() < P_kill_M:
                nodes_to_kill.append(node_id)

        for node_id in set(nodes_to_kill):
            self.G.nodes[node_id]['is_alive'] = False

    def run_simulation(self, t_end):
        """Main simulation loop."""
        t_eval = np.arange(self.current_time, t_end + self.dt, self.dt)

        T_surface_count = 0
        N_alive = 0

        for t in t_eval:
            self.current_time = t

            living_nodes = self.get_living_nodes()
            N_alive = len(living_nodes)

            if N_alive == 0:
                T_surface_count = 0
                self.step_sde_globals(T_surface_count)
            else:
                surface_nodes, _ = self.get_surface_nodes(living_nodes)
                T_surface_count = len(surface_nodes)

                # 1. Update global E and M
                self.step_sde_globals(T_surface_count)

                # 2. Apply tumor killing
                self.step_tumor_killing(living_nodes, surface_nodes)

                # 3. Apply tumor proliferation
                living_nodes_after_kill = self.get_living_nodes()
                self.step_tumor_proliferation(living_nodes_after_kill, self.params['K_cap'])

            # Record history
            # --- Added history recording ---
            self.history.append([t, self.E, N_alive, self.M, T_surface_count])

        return np.array(self.history)

# ==============================================================================
# VISUALIZATION AND MAIN EXECUTION
# ==============================================================================

def plot_ode_results(t, y):
    E, T, M = y
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Cell Population')
    ax1.plot(t, E, 'g-', label='E (Effector)')
    ax1.plot(t, T, 'r-', label='T (Tumor)')
    ax1.set_ylim(0, max(1, np.max(E), np.max(T)) * 1.1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Drug Concentration')
    ax2.plot(t, M, 'b--', label='M (Chemo)')
    ax2.set_ylim(0, max(1, np.max(M) * 1.1))

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.title('ODE Model (orig)')
    plt.show()

def plot_sde_ensemble(t, results):
    E_runs = results[:, :, 0]
    T_runs = results[:, :, 1]
    M_runs = results[:, :, 2]

    E_mean, E_std = np.mean(E_runs, axis=0), np.std(E_runs, axis=0)
    T_mean, T_std = np.mean(T_runs, axis=0), np.std(T_runs, axis=0)

    fig, ax = plt.subplots()
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cell Population')

    ax.plot(t, E_mean, 'g-', label='E (Effector) - Mean')
    ax.plot(t, T_mean, 'r-', label='T (Tumor) - Mean')

    ax.fill_between(t, E_mean - E_std, E_mean + E_std, color='g', alpha=0.2)
    ax.fill_between(t, T_mean - T_std, T_mean + T_std, color='r', alpha=0.2)

    plt.legend()
    plt.title(f'SDE Model (stochastic) - {len(results)} Runs')
    plt.show()

def plot_hybrid_timeseries(history):
    if history.size == 0:
        print("Hybrid simulation history is empty.")
        return

    t, E, N_alive, M, T_surface = history.T

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Cell Population')
    ax1.plot(t, E, 'g-', label='E (Global Effector)')
    ax1.plot(t, N_alive, 'r-', label='T (Total Living Cells)')
    ax1.plot(t, T_surface, 'r:', label='T (Surface Cells)')
    ax1.set_ylim(0, max(1, np.max(E), np.max(N_alive)) * 1.1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Drug Concentration')
    ax2.plot(t, M, 'b--', label='M (Global Chemo)')
    ax2.set_ylim(0, max(1, np.max(M) * 1.1))

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.title('Hybrid Network Model (network)')
    plt.show()

if __name__ == "__main__":
    # --- Shared Simulation Parameters ---
    y0_E = 3000 #30000
    y0_T = 4000 #40000                          --------- CHANGE
    y0_M = 0
    # ---  Initialized state vector ---
    y0_ode_sde = [y0_E, y0_T, y0_M]

    t_span = (0, 50)
    t_eval_ode = np.linspace(t_span[0], t_span[1], 100)
    dt_sde_net = 0.1 # Time step for stochastic models

    # --- Scenario: Low growth (Fig 1a) ---
    print("Running Scenario: Low Growth (r=0.00431, V=0.5)")
    params_low_g = get_base_parameters()
    params_low_g['r'] = 4.31e-3
    params_low_g['V'] = 0.5

    # 1. Run ODE Model
    t_ode, y_ode = run_ode_simulation(params_low_g, y0_ode_sde, t_span, t_eval_ode)
    plot_ode_results(t_ode, y_ode)

    # 2. Run SDE Model
    t_sde, y_sde_runs = run_sde_simulation(params_low_g, y0_ode_sde, t_span, dt_sde_net, num_runs=20)
    plot_sde_ensemble(t_sde, y_sde_runs)

    # 3. Run Network Model
    params_net = params_low_g.copy()
    params_net['network_radius'] = 0.15 # Controls network density

    G = generate_tumor_network(n_nodes=y0_T, radius=params_net['network_radius'])
    y0_globals_net = [y0_E, y0_M]

    simulator = HybridSimulator(y0_globals_net, G, params_net, dt_sde_net)
    history = simulator.run_simulation(t_span[1])
    plot_hybrid_timeseries(history)

    # --- Scenario: High growth (Fig 1b/1c) ---
    print("Running Scenario: High Growth (r=0.47, V=0.6)")
    params_high_g = get_base_parameters()
    params_high_g['r'] = 0.47
    params_high_g['V'] = 0.6 # From Fig 1c
    params_high_g['K_cap'] = 1 / params_high_g['b']

    # 1. Run ODE Model
    t_ode_hg, y_ode_hg = run_ode_simulation(params_high_g, y0_ode_sde, t_span, t_eval_ode)
    plot_ode_results(t_ode_hg, y_ode_hg)

    # ... SDE and Network runs for this scenario would follow the same pattern...
    print("--- High Growth SDE Run ---")
    t_sde_hg, y_sde_runs_hg = run_sde_simulation(params_high_g, y0_ode_sde, t_span, dt_sde_net, num_runs=20)
    plot_sde_ensemble(t_sde_hg, y_sde_runs_hg)

    print("--- High Growth Network Run ---")
    params_net_hg = params_high_g.copy()
    params_net_hg['network_radius'] = 0.15

    G_hg = generate_tumor_network(n_nodes=y0_T, radius=params_net_hg['network_radius'])

    simulator_hg = HybridSimulator(y0_globals_net, G_hg, params_net_hg, dt_sde_net)
    history_hg = simulator_hg.run_simulation(t_span[1])
    plot_hybrid_timeseries(history_hg)

    print("Simulation complete.")