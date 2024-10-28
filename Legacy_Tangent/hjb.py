import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 2  # CRRA utility parameter
r = 0.03  # interest rate
rho = 0.05  # discount rate
z1 = 0.1
z2 = 0.2
z = np.array([z1, z2])
la1 = 0.02  # lambda_1
la2 = 0.03  # lambda_2
la = np.array([la1, la2])

# State space discretization
I = 500
amin = -0.02  # borrowing constraint
amax = 2
a = np.linspace(amin, amax, I)
da = (amax - amin) / (I - 1)

# Initialize variables
v = np.zeros((I, 2))  # value function
v[:, 0] = (z[0] + r * a) ** (1 - s) / (1 - s) / rho
v[:, 1] = (z[1] + r * a) ** (1 - s) / (1 - s) / rho
dVf = np.zeros((I, 2))
dVb = np.zeros((I, 2))
c = np.zeros((I, 2))

maxit = 20000
crit = 1e-6
dist = []

for n in range(maxit):
    V = v.copy()
    # Forward difference
    dVf[:-1, :] = (V[1:, :] - V[:-1, :]) / da
    dVf[-1, :] = 0  # will never be used
    # Backward difference
    dVb[1:, :] = (V[1:, :] - V[:-1, :]) / da
    dVb[0, :] = (z + r * amin) ** (-s)  # state constraint boundary condition

    # Consumption and savings with forward and backward differences
    cf = dVf ** (-1 / s)
    muf = z + r * a[:, None] - cf
    cb = dVb ** (-1 / s)
    mub = z + r * a[:, None] - cb
    c0 = z + r * a[:, None]
    dV0 = c0 ** (-s)

    # Upwind scheme
    If = muf > 0  # positive drift -> forward difference
    Ib = mub < 0  # negative drift -> backward difference
    I0 = ~(If | Ib)  # at steady state
    Ib[-1, :] = 1; If[-1, :] = 0  # ensure backward difference at amax

    dV_Upwind = dVf * If + dVb * Ib + dV0 * I0
    c = dV_Upwind ** (-1 / s)
    V_switch = np.column_stack((V[:, 1], V[:, 0]))
    Vchange = c ** (1 - s) / (1 - s) + dV_Upwind * (z + r * a[:, None] - c) + la * (V_switch - V) - rho * V

    # CFL condition
    Delta = 0.9 * da / np.max(z2 + r * a)
    v += Delta * Vchange

    # Convergence check
    max_diff = np.max(np.abs(Vchange))
    dist.append(max_diff)
    if max_diff < crit:
        print(f'Value Function Converged, Iteration = {n + 1}')
        break

# Plot convergence
plt.figure()
plt.plot(dist)
plt.xlabel('Iteration')
plt.ylabel('||V^{n+1} - V^n||')
plt.grid(True)
plt.title('Convergence of Value Function')
plt.show()

# Plot value function
plt.figure()
plt.plot(a, v[:, 0], label='V_1(a)')
plt.plot(a, v[:, 1], label='V_2(a)')
plt.xlabel('a')
plt.ylabel('V_i(a)')
plt.grid(True)
plt.legend()
plt.title('Value Function')
plt.show()

# Plot consumption policy
plt.figure()
plt.plot(a, c[:, 0], label='c_1(a)')
plt.plot(a, c[:, 1], label='c_2(a)')
plt.xlabel('a')
plt.ylabel('c_i(a)')
plt.grid(True)
plt.legend()
plt.title('Optimal Consumption Policy')
plt.show()

# Plot drift (adot)
adot = z + r * a[:, None] - c
plt.figure()
plt.plot(a, adot[:, 0], label='adot_1(a)')
plt.plot(a, adot[:, 1], label='adot_2(a)')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('a')
plt.ylabel('s_i(a)')
plt.grid(True)
plt.legend()
plt.title('Drift of Wealth (adot)')
plt.show()
