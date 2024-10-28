import numpy as np
from collections import deque

# ------------------------- User Input Section -------------------------

# Time periods
TIME_PERIODS = [0, 1, 2]  # Assuming two periods: 0 and 1

# Define policy actions and their indices
DECISIONS = ['LowTax_Welfare', 'LowTax_Subsidy', 'HighTax_Welfare', 'HighTax_Subsidy']
DECISION_INDICES = {name: idx for idx, name in enumerate(DECISIONS)}

# Initial conditions (corrected)
initial_state = {
    'E': 125,    # Calculated initial emissions
    'W': 21050   # Calculated initial welfare
}

# Parameters (adjusted c_tax)
PARAMETERS = {
    'A_B': 10,        # Productivity of Black Firms
    'A_G': 8,         # Productivity of Green Firms
    'alpha_B': 0.7,   # Labor share parameter for Black Firms
    'alpha_G': 0.7,   # Labor share parameter for Green Firms
    'w_B': 15,        # Wage rate for Black Workers
    'w_G': 15,        # Wage rate for Green Workers
    'epsilon': 0.5,   # Emissions per unit of Black Firm output
    'c_tax': 200,     # Increased Carbon tax rate
    'P_B': 50,        # Price of Black Firm output
    'P_G': 50,        # Price of Green Firm output
    'lambda': 0.05,   # Subsidy efficiency
    'omega': 1.0,     # Welfare distribution efficiency
    'psi': 1.0        # Weight on final emissions in objective function
}

# Utility function (for policy maker)
def utility_function(W, E, psi):
    # Define the utility based on aggregate welfare and emissions
    utility = W - psi * E
    return utility

# ------------------------- End of User Input Section -------------------------

# State node representation
class StateNode:
    def __init__(self, time, state_vars, decisions=None, parent=None):
        self.time = time
        self.state_vars = state_vars.copy()
        self.decisions = decisions  # Decisions taken to reach this state
        self.parent = parent        # Parent StateNode
        self.children = []          # Child StateNodes

    def __repr__(self):
        state_vars_str = ', '.join([f"{k}: {v:.2f}" for k, v in self.state_vars.items()])
        decisions_str = ', '.join([f"{DECISIONS[i]}" for i, d in enumerate(self.decisions) if d == 1])
        return (f"Time: {self.time}, {state_vars_str}, Decisions: {decisions_str}")

# State transition function
def state_transition(state_vars_prev, decisions, params):
    state_vars = state_vars_prev.copy()
    D = decisions

    # Extract parameters
    A_B = params['A_B']
    A_G = params['A_G']
    alpha_B = params['alpha_B']
    alpha_G = params['alpha_G']
    w_B = params['w_B']
    w_G = params['w_G']
    epsilon = params['epsilon']
    c_tax = params['c_tax']
    P_B = params['P_B']
    P_G = params['P_G']
    lambda_ = params['lambda']
    omega = params['omega']

    # Map decisions to policy actions
    executed_action = None
    for idx, d in enumerate(D):
        if d == 1:
            executed_action = DECISIONS[idx]
            break

    # Determine carbon tax level and tax distribution based on the action
    if executed_action == 'LowTax_Welfare':
        c_t = 0
        d_t = 0
    elif executed_action == 'LowTax_Subsidy':
        c_t = 0
        d_t = 1
    elif executed_action == 'HighTax_Welfare':
        c_t = c_tax
        d_t = 0
    elif executed_action == 'HighTax_Subsidy':
        c_t = c_tax
        d_t = 1
    else:
        c_t = 0
        d_t = 0

    # Calculate Black Firms' effective cost
    effective_cost_B = w_B + c_t * epsilon / A_B

    # Black Firms' labor demand
    L_B = ((P_B * A_B * alpha_B) / effective_cost_B) ** (1 / (1 - alpha_B))

    # Black Firms' output
    Q_B = A_B * L_B ** alpha_B

    # Emissions
    E_t = epsilon * Q_B

    # Calculate tax revenue
    T = c_t * E_t

    # Calculate Green Firms' effective productivity
    if d_t == 1:
        # Subsidies to Green Firms
        A_G_effective = A_G + lambda_ * T
    else:
        A_G_effective = A_G

    # Green Firms' labor demand
    L_G = ((P_G * A_G_effective * alpha_G) / w_G) ** (1 / (1 - alpha_G))

    # Green Firms' output
    Q_G = A_G_effective * L_G ** alpha_G

    # Welfare distribution to Black Workers
    if d_t == 0:
        Welfare_BW = omega * T
    else:
        Welfare_BW = 0

    # Workers' wage income
    W_income_BW = w_B * L_B
    W_income_GW = w_G * L_G

    # Workers' utility
    U_BW = W_income_BW + Welfare_BW
    U_GW = W_income_GW

    # Firms' profits
    Profit_B = P_B * Q_B - w_B * L_B - c_t * E_t
    Profit_G = P_G * Q_G - w_G * L_G

    # Aggregate welfare
    W_t = U_BW + U_GW + Profit_B + Profit_G

    # Update state variables
    state_vars['E'] = E_t
    state_vars['W'] = W_t

    # Store additional variables if needed
    state_vars['U_BW'] = U_BW
    state_vars['U_GW'] = U_GW
    state_vars['L_B'] = L_B
    state_vars['L_G'] = L_G
    state_vars['Q_B'] = Q_B
    state_vars['Q_G'] = Q_G
    state_vars['Profit_B'] = Profit_B
    state_vars['Profit_G'] = Profit_G

    return state_vars

# Build the state-transition graph
def build_graph(initial_state, params):
    root = StateNode(time=0, state_vars=initial_state)
    queue = deque([root])
    paths = []

    while queue:
        current_node = queue.popleft()

        if current_node.time == max(TIME_PERIODS):
            # Backtrack to get the path
            path = []
            node = current_node
            while node.parent is not None:
                path.append({
                    'time': node.time,
                    'decisions': node.decisions,
                    'state_vars': node.state_vars
                })
                node = node.parent
            path.reverse()
            paths.append(path)
            continue

        # Generate all possible policy actions
        action_combinations = []
        for idx in range(len(DECISIONS)):
            decisions = [0] * len(DECISIONS)
            decisions[idx] = 1
            action_combinations.append(decisions)

        # For each action, create the next state
        for decisions in action_combinations:
            # Simulate state transition
            state_vars_next = state_transition(
                current_node.state_vars,
                decisions,
                params
            )

            # Create child node
            child_node = StateNode(
                time=current_node.time + 1,
                state_vars=state_vars_next,
                decisions=decisions,
                parent=current_node
            )
            current_node.children.append(child_node)
            queue.append(child_node)

    return paths

# Execute the graph building
paths = build_graph(initial_state, PARAMETERS)

# Process and sort the paths
results = []

for path in paths:
    # Initialize cumulative welfare and emissions
    cumulative_W = initial_state['W']
    cumulative_E = initial_state['E']

    # Go through each step in the path
    for step in path:
        state_vars = step['state_vars']
        cumulative_W += state_vars['W']
        cumulative_E += state_vars['E']

    # Calculate total utility
    total_utility = utility_function(cumulative_W, cumulative_E, PARAMETERS['psi'])

    results.append({
        'path': path,
        'total_utility': total_utility,
        'cumulative_emissions': cumulative_E,
        'cumulative_welfare': cumulative_W
    })

# Sort paths based on total utility (higher is better) and cumulative emissions (lower is better)
results.sort(key=lambda x: (-x['total_utility'], x['cumulative_emissions']))

# Display the top paths
print(f"Total feasible paths: {len(results)}\n")

for idx, result in enumerate(results):
    print(f"Path {idx + 1}:")
    path = result['path']
    for step in path:
        time = step['time']
        decisions = step['decisions']
        state_vars = step['state_vars']
        decision_names = [DECISIONS[i] for i, d in enumerate(decisions) if d == 1]
        state_vars_str = ', '.join([f"{k}: {v:.2f}" for k, v in state_vars.items() if k in ['E', 'W']])
        print(f"  Time {time}: Decisions: {decision_names}, {state_vars_str}")
    print(f"  Total Utility: {result['total_utility']:.2f}")
    print(f"  Cumulative Emissions: {result['cumulative_emissions']:.2f}")
    print(f"  Cumulative Welfare: {result['cumulative_welfare']:.2f}\n")
