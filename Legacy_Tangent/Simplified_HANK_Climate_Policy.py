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
delta = 0.1          # Productivity effect of carbon intensity
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
model.C0 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.C1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.L0 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.L1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, 1), initialize=0.5)
model.K1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.01, None), initialize=1.0)
model.buy_new_capital = pyo.Var(domain=pyo.Binary, initialize=0)

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

# Production function
def Y_t(A_t, K_t, L_t, eta_t):
    return A_t * (1 + delta * eta_t) * K_t ** alpha * L_t ** (1 - alpha)

# Objective function: maximize expected utility over two periods
def objective_rule(model):
    expected_utility = 0
    for A0, eta0, A1, eta1 in zip(A_samples, eta_samples, A_samples, eta_samples):
        utility_old = u(model.C0, model.L0, eta0) + beta * u(model.C1, model.L1, eta1)
        Y0_old = Y_t(A0, K0, model.L0, eta0)
        Y1_old = Y_t(A1, model.K1, model.L1, eta1)
        
        utility_new = u(model.C0, model.L0, eta1) + beta * u(model.C1, model.L1, eta1)
        Y0_new = Y_t(A1, K0, model.L0, eta1)
        Y1_new = Y_t(A1, model.K1, model.L1, eta1)
        
        expected_utility += (1 - model.buy_new_capital) * (utility_old - tau * (Y0_old * eta0 + beta * Y1_old * eta1)) + \
                            model.buy_new_capital * (utility_new - tau * (Y0_new * eta1 + beta * Y1_new * eta1))
    
    return -expected_utility / num_samples

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints
def budget_constraint_0(model):
    expected_Y0 = sum(Y_t(A, K0, model.L0, eta) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_eta0 = sum(eta_samples) / num_samples
    return (model.C0 + model.K1 == 
            expected_Y0 * (1 - tau * expected_eta0) + 
            (1 - dep_rate) * K0 + T_t + epsilon_t - 
            model.buy_new_capital * (model.K1 - (1 - dep_rate) * K0))

def budget_constraint_1(model):
    expected_Y1 = sum(Y_t(A, model.K1, model.L1, eta) for A, eta in zip(A_samples, eta_samples)) / num_samples
    expected_eta1 = sum(eta_samples) / num_samples
    return (model.C1 == 
            expected_Y1 * (1 - tau * expected_eta1) + 
            (1 - dep_rate) * model.K1 + T_t + epsilon_t)

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
    print(f"C0 = {pyo.value(model.C0):.4f}")
    print(f"C1 = {pyo.value(model.C1):.4f}")
    print(f"L0 = {pyo.value(model.L0):.4f}")
    print(f"L1 = {pyo.value(model.L1):.4f}")
    print(f"K1 = {pyo.value(model.K1):.4f}")
    print(f"Buy new capital = {'Yes' if pyo.value(model.buy_new_capital) > 0.5 else 'No'}")
    
    # Calculate expected wages and interest rates
    expected_A = np.mean(A_samples)
    expected_eta = np.mean(eta_samples)
    w0 = (1 - alpha) * Y_t(expected_A, K0, pyo.value(model.L0), expected_eta) / pyo.value(model.L0)
    w1 = (1 - alpha) * Y_t(expected_A, pyo.value(model.K1), pyo.value(model.L1), expected_eta) / pyo.value(model.L1)
    r0_calc = alpha * Y_t(expected_A, K0, pyo.value(model.L0), expected_eta) / K0 - dep_rate
    r1_calc = alpha * Y_t(expected_A, pyo.value(model.K1), pyo.value(model.L1), expected_eta) / pyo.value(model.K1) - dep_rate
    
    print(f"Expected w0 = {w0:.4f}")
    print(f"Expected w1 = {w1:.4f}")
    print(f"Expected r0 (calculated) = {r0_calc:.4f}")
    print(f"Expected r1 (calculated) = {r1_calc:.4f}")

    # Validation checks
    print("\nValidation Checks:")
    budget_check_0 = pyo.value(model.C0 + model.K1 - (w0 * model.L0 + (1 + r0_calc - dep_rate) * K0 + T_t + epsilon_t - model.buy_new_capital * (model.K1 - (1 - dep_rate) * K0)))
    budget_check_1 = pyo.value(model.C1 - (w1 * model.L1 + (1 + r1_calc - dep_rate) * model.K1 + T_t + epsilon_t))
    print(f"Budget Constraint Period 0: {budget_check_0:.6f} (should be close to 0)")
    print(f"Budget Constraint Period 1: {budget_check_1:.6f} (should be close to 0)")

else:
    print("No optimal solution found.")
    print("Solver Status:", result.solver.status)
    print("Termination Condition:", result.solver.termination_condition)
