import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from scipy.stats import multivariate_normal

# Define parameters
beta = 0.96          # Discount factor
sigma = 2            # CRRA coefficient
gamma = 0.5          # Preference parameter for carbon intensity
kappa = 1            # Preference parameter
alpha = 0.3          # Capital share in production
delta_high = 0.1     # Productivity effect of carbon intensity for high-carbon firms
delta_low = -0.05    # Productivity effect for low-carbon firms
tau = 10             # Emission tax
K0 = 1               # Initial capital
chi = 1              # Labor disutility parameter
nu = 1               # Labor supply elasticity
P_t = 1              # Price level (normalized)
T_t = 0              # Government transfers
epsilon_t = 0        # Income shock (assuming zero for simplicity)
dep_rate = 0.1       # Depreciation rate
r0 = 0.02            # Interest rate for period 0
r1 = 0.02            # Interest rate for period 1

# Parameters for joint distribution of A_bar and eta
mu_A = 1.0           # Mean of A_bar
mu_eta = 0.5         # Mean of eta
sigma_A = 0.2        # Standard deviation of A_bar
sigma_eta = 0.1      # Standard deviation of eta
rho = 0.5            # Correlation coefficient

# Create the model
model = pyo.ConcreteModel()

# Define variables
model.C0_low = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.C0_high = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.C1_low = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.C1_high = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.L0_low = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.L0_high = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.L1_low = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.L1_high = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.K1_low = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.K1_high = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)

# Generate samples for A and eta
num_samples = 1000
mean = [mu_A, mu_eta]
cov = [[sigma_A**2, rho*sigma_A*sigma_eta],
       [rho*sigma_A*sigma_eta, sigma_eta**2]]
samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)

A_samples = samples[:, 0]
eta_samples = np.clip(samples[:, 1], 0, 1)  # Ensure eta is between 0 and 1

# Utility function
def u(C_t, L_t, eta_t):
    return (C_t ** (1 - sigma) / (1 - sigma)) * (1 - gamma * eta_t) ** kappa - chi * L_t ** (1 + nu) / (1 + nu)

# Production function for high and low carbon firms
def Y_t(A_t, K_t, L_t, eta_t, delta):
    return A_t * (1 + delta * eta_t) * K_t ** alpha * L_t ** (1 - alpha)

# Objective function: maximize expected utility over two periods
def objective_rule(model):
    expected_utility = 0
    for A0, eta0, A1, eta1 in zip(A_samples, eta_samples, A_samples, eta_samples):
        utility_low = (
            u(model.C0_low, model.L0_low, eta0) + 
            beta * u(model.C1_low, model.L1_low, eta1)
        )
        utility_high = (
            u(model.C0_high, model.L0_high, eta0) + 
            beta * u(model.C1_high, model.L1_high, eta1)
        )
        
        Y0_low = Y_t(A0, K0, model.L0_low, eta0, delta_low)
        Y1_low = Y_t(A1, model.K1_low, model.L1_low, eta1, delta_low)
        
        Y0_high = Y_t(A0, K0, model.L0_high, eta0, delta_high)
        Y1_high = Y_t(A1, model.K1_high, model.L1_high, eta1, delta_high)
        
        expected_utility += (
            utility_low - tau * (Y0_low * eta0 + beta * Y1_low * eta1) +
            utility_high - tau * (Y0_high * eta0 + beta * Y1_high * eta1)
        )
    
    return -expected_utility / num_samples

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints
def budget_constraint_0(model):
    expected_Y0_low = sum(Y_t(A, K0, model.L0_low, eta, delta_low) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_Y0_high = sum(Y_t(A, K0, model.L0_high, eta, delta_high) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_eta0 = sum(eta_samples) / num_samples
    return (model.C0_low + model.C0_high + model.K1_low + model.K1_high == 
            expected_Y0_low * (1 - tau * expected_eta0) + 
            expected_Y0_high * (1 - tau * expected_eta0) +
            (1 - dep_rate) * (K0 + K0) + T_t + epsilon_t)

def budget_constraint_1(model):
    expected_Y1_low = sum(Y_t(A, model.K1_low, model.L1_low, eta, delta_low) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_Y1_high = sum(Y_t(A, model.K1_high, model.L1_high, eta, delta_high) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_eta1 = sum(eta_samples) / num_samples
    return (model.C1_low + model.C1_high == 
            expected_Y1_low * (1 - tau * expected_eta1) + 
            expected_Y1_high * (1 - tau * expected_eta1) +
            (1 - dep_rate) * (model.K1_low + model.K1_high) + T_t + epsilon_t)

model.budget_constraint_0 = pyo.Constraint(rule=budget_constraint_0)
model.budget_constraint_1 = pyo.Constraint(rule=budget_constraint_1)

# Solve the model
solver = SolverFactory('ipopt')
solver.options['linear_solver'] = 'mumps'
solver.options['max_iter'] = 5000
solver.options['tol'] = 1e-6
result = solver.solve(model, tee=True)

# Check if the solver found a solution
if result.solver.status == pyo.SolverStatus.ok and result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\nOptimal Solution Found:")
    print(f"C0_low = {pyo.value(model.C0_low):.4f}")
    print(f"C0_high = {pyo.value(model.C0_high):.4f}")
    print(f"C1_low = {pyo.value(model.C1_low):.4f}")
    print(f"C1_high = {pyo.value(model.C1_high):.4f}")
    print(f"L0_low = {pyo.value(model.L0_low):.4f}")
    print(f"L0_high = {pyo.value(model.L0_high):.4f}")
    print(f"L1_low = {pyo.value(model.L1_low):.4f}")
    print(f"L1_high = {pyo.value(model.L1_high):.4f}")
    print(f"K1_low = {pyo.value(model.K1_low):.4f}")
    print(f"K1_high = {pyo.value(model.K1_high):.4f}")
else:
    print("No optimal solution found.")
    print("Solver Status:", result.solver.status)
    print("Termination Condition:", result.solver.termination_condition)
