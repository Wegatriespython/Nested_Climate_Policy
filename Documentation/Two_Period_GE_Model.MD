# Two-Period General Equilibrium Model with Portfolio Choice and Diminishing Returns

## Introduction

This document presents a two-period General Equilibrium (GE) model that incorporates climate policy extensions, portfolio choice over a spectrum of technologies, and diminishing marginal returns to capital. Firms choose how to allocate their investment across various technologies, each characterized by different levels of productivity and emissions intensity. The inclusion of diminishing marginal returns prevents firms from investing exclusively in a single technology, promoting a diversified technology adoption that more accurately reflects real-world scenarios.

---

## Variables

- **Consumption:**
  - \( C_0 \), \( C_1 \): Consumption in periods 0 and 1.
- **Labor Supply:**
  - \( L_0 \), \( L_1 \): Labor supply in periods 0 and 1.
- **Investment:**
  - \( I_0 \), \( I_1 \): Total investment in periods 0 and 1.
- **Capital Stock:**
  - \( K_0 \), \( K_1 \): Total capital stock in periods 0 and 1.
- **Capital Allocation:**
  - \( K_t^i \): Capital stock allocated to technology \( i \) in period \( t \).
  - \( \theta_t^i \): Share of total capital stock allocated to technology \( i \) in period \( t \).
- **Technology Levels:**
  - \( A_t^i \): Productivity level of technology \( i \) in period \( t \).
- **Emissions Intensity:**
  - \( \eta_t^i \): Emissions intensity of technology \( i \) in period \( t \).
- **Effective Technology and Emissions Intensity:**
  - \( A_{eff,t} \): Effective aggregate productivity in period \( t \).
  - \( \eta_{eff,t} \): Effective aggregate emissions intensity in period \( t \).
- **Interest Rates:**
  - \( r_0 \), \( r_1 \): Interest rates in periods 0 and 1.
- **Wages:**
  - \( w_0 \), \( w_1 \): Wages in periods 0 and 1.
- **Adjustment Costs:**
  - \( \Delta \eta_t \): Change in effective emissions intensity in period \( t \).
  - \( \text{AdjCost}_t \): Adjustment cost in period \( t \).
- **Labor Efficiency:**
  - \( \text{Eff}_L(t) \): Labor efficiency in period \( t \).

## Additional Variable Specifications

### Technology Distribution Parameters
- $\mu_A$: Mean productivity across technologies
- $\mu_\eta$: Mean emissions intensity
- $\sigma_A$: Standard deviation of productivity
- $\sigma_\eta$: Standard deviation of emissions intensity
- $\rho$: Correlation coefficient between productivity and emissions intensity

The pair $(A_t,\eta_t)$ follows a joint distribution with probability density function:
$f(A_t,\eta_t)=f_{A,\eta}(A_t,\eta_t)$

### Worker Skill Distribution
- $\theta \in [\underline{\theta},\overline{\theta}]$: Worker skill level
- $F(\theta)$: Cumulative distribution function of skills
- $\psi(\theta)$: Function decreasing in $\theta$ (higher skills imply lower adjustment costs)

---

## Parameters

- \( \alpha \in (0,1) \): Capital share in production.
- \( \beta \in (0,1) \): Discount factor.
- \( \sigma > 0 \): Relative risk aversion.
- \( \delta \in (0,1) \): Depreciation rate.
- \( \chi > 0 \): Disutility of labor parameter.
- \( \nu \geq 0 \): Inverse of labor supply elasticity.
- \( \gamma \geq 0 \): Adjustment cost parameter.
- \( \tau_0, \tau_1 \geq 0 \): Carbon taxes in periods 0 and 1.
- \( \phi > 0 \): Diminishing returns parameter.
- \( K_{init} > 0 \): Initial capital stock.
- \( A_{init} \), \( \eta_{init} \): Initial technology productivity and emissions intensity.
- **Technology Set:**
  - \( \{ A_t^i, \eta_t^i \} \): Set of available technologies in period \( t \).

## Extended Parameter Specifications

### Technology Parameters
- $\phi > 0$: Diminishing returns parameter for technology-specific capital
- $\rho \in [-1,1]$: Elasticity of substitution between different types of capital
- $skill\_factor$: Scaling parameter for skill-based adjustment costs

### Adjustment Cost Parameters
- $\gamma_{tech}$: Technology-specific adjustment cost parameter
- $\gamma_{labor}$: Labor market adjustment cost parameter

---

## Equations

### 1. Capital Evolution

#### Total Capital Stock:
- **Period 0:**
  \[
  K_0 = K_{init} + I_0 - \delta K_{init}
  \]

- **Period 1:**
  \[
  K_1 = K_0 + I_1 - \delta K_0
  \]

**Note on Investment Timing:** In this model, investment affects capital stock immediately within the same period, rather than following the traditional time-to-build lag. This design choice allows for active capital allocation decisions in both periods and enables the model to adjust initial capital allocation across technologies. This immediate effect is appropriate given our focus on portfolio choice and technology transition dynamics.

### 2. Capital Allocation

Firms allocate capital across technologies \( i \) in each period \( t \):

\[
K_t = \sum_i K_t^i
\]

Capital allocation shares:

\[
\theta_t^i = \frac{K_t^i}{K_t}, \quad \sum_i \theta_t^i = 1, \quad \theta_t^i \geq 0
\]

### 3. Effective Technology Levels

Aggregate effective productivity and emissions intensity are weighted averages:

\[
A_{eff,t} = \sum_i \theta_t^i A_t^i
\]

\[
\eta_{eff,t} = \sum_i \theta_t^i \eta_t^i
\]

### 4. Production Function with Diminishing Returns

The production function should be modified to apply emissions tax at the technology-specific level for consistency:

\[
Y_t = \sum_i (1 - \tau_t \eta_t^i) A_t^i (K_t^i)^\alpha (\text{Eff}_L(t) L_t)^{1 - \alpha}
\]

This ensures consistent treatment of emissions taxation across production and factor prices.

### 5. Labor Efficiency and Capital Adjustment Costs

The model separates frictions into two distinct mechanisms:

1. **Labor Market Frictions (Labor Efficiency):**
First, we define the skill factor as the average inverse skill level in the economy:

\[
\text{skill\_factor} = \int_{\underline{\theta}}^{\overline{\theta}} \frac{1}{\theta} dF(\theta)
\]

Then, labor efficiency is affected by changes in emissions intensity, moderated by this skill factor:

\[
\Delta \eta_t = \eta_{eff,t} - \eta_{eff,t-1}
\]

\[
\text{Eff}_L(t) = \frac{1}{1 + \gamma_{labor} (\Delta \eta_t)^2 \times \text{skill\_factor}}
\]

Where:
- $\text{skill\_factor}$ represents the aggregate effect of skill distribution on adjustment costs
- Higher aggregate skills (lower skill_factor) lead to smaller efficiency losses
- Larger technology changes (larger $\Delta \eta_t$) cause greater efficiency losses

2. **Capital Market Frictions (Resource Costs):**
Capital adjustment costs represent the direct resource costs of changing technology:

\[
\text{AdjCost}_t = \gamma_{tech} \sum_i (\theta_t^i - \theta_{t-1}^i )^2
\]

These costs represent physical resources used in retooling and reconfiguring capital stock.

### 6. Factor Prices

#### Interest Rates:
The marginal product of capital for technology $i$ is:

\[
MPK_t^i = \alpha A_t^i (K_t^i)^{\alpha - 1} (\text{Eff}_L(t) L_t)^{1 - \alpha}
\]

After accounting for the emissions tax:

\[
r_t^i = (1 - \tau_t \eta_t^i)MPK_t^i - \delta
\]

\[
r_t = \sum_i \theta_t^i r_t^i
\]
#### Wages:
The wage rate becomes:

\[
w_t = (1 - \alpha) \sum_i (1 - \tau_t \eta_t^i) \frac{A_t^i (K_t^i)^\alpha (\text{Eff}_L(t) L_t)^{-\alpha}}{\text{Eff}_L(t)}
\]

### 7. Budget Constraints

The budget constraints reflect only the direct capital adjustment costs, as labor efficiency effects are already captured in the production function:

#### Period 0:
\[
C_0 + I_0 + \text{AdjCost}_0 = Y_0
\]

Where $Y_0$ already incorporates labor efficiency effects through $\text{Eff}_L(0)$

#### Period 1:
\[
C_1 + I_1 + \text{AdjCost}_1 = Y_1 + (1-\delta)K_1
\]

Where $Y_1$ already incorporates labor efficiency effects through $\text{Eff}_L(1)$

### 8. Euler Equation

The Euler equation needs to reflect the net return including terminal value:

\[
\beta \left( \frac{C_1}{C_0} \right)^{-\sigma} = 1 + r_1 + (1-\delta)
\]

**Note:** The return now includes both the productive return $r_1$ and the terminal value of undepreciated capital $(1-\delta)$.

### 9. Labor Supply First-Order Conditions

Households optimize labor supply:

\[
\chi L_t^\nu = C_t^{ -\sigma } w_t
\]

### 10. Household Utility Maximization

The representative household maximizes utility:

\[
U = \frac{ C_0^{ 1 - \sigma } }{ 1 - \sigma } - \chi \frac{ L_0^{ 1 + \nu } }{ 1 + \nu } + \beta \left( \frac{ C_1^{ 1 - \sigma } }{ 1 - \sigma } - \chi \frac{ L_1^{ 1 + \nu } }{ 1 + \nu } \right)
\]

### 11. Market Clearing Conditions

#### Goods Market:
Period 0:
\[
Y_0 = C_0 + I_0 + \text{AdjCost}_0
\]

Period 1:
\[
Y_1 + (1-\delta)K_1 = C_1 + I_1 + \text{AdjCost}_1
\]

#### Labor Market:
\[
L_t^{demand} = L_t^{supply}
\]

\[
    L_t^{demand} = \left[\frac{w_t}{(1-\alpha)\sum_i (1-\tau_t\eta_t^i)A_t^i(K_t^i)^\alpha}\right]^{-\frac{1}{\alpha}}\text{Eff}_L(t)
    \]

#### Capital Market:

For each technology $i$:
\[K_t^i = \theta_t^i K_t\]
\[\sum_i \theta_t^i = 1\]

---

## Additional Constraints and Conditions

- **Non-negativity and Feasibility:**

  \[
  K_t^i \geq 0, \quad \theta_t^i \geq 0, \quad \forall i, t
  \]

  \[
  L_t \geq 0, \quad C_t \geq 0, \quad I_t \geq - \delta K_{t-1 }
  \]

- **Initial Conditions:**

  \[
  K_{-1} = K_{init}, \quad \eta_{eff, -1 } = \eta_{init}
  \]

---

## Notes on Diminishing Marginal Returns

Including diminishing marginal returns to each type of capital prevents the firm from allocating all capital to a single technology. The exponent \( \alpha < 1 \) in \( (K_t^i )^\alpha \) ensures that the marginal productivity decreases as more capital is allocated to technology \( i \).

---

## Alternative Production Function Specification

Alternatively, a CES (Constant Elasticity of Substitution) production function can be used to allow for substitution between different types of capital:

\[
Y_t = (1 - \tau_t \eta_{eff,t} ) \left( \sum_i \left( A_t^i K_t^i \right)^{ \rho } \right)^{ \frac{ \alpha }{ \rho } } ( \text{Eff}_L(t) L_t )^{ 1 - \alpha }
\]

- \( \rho \) determines the elasticity of substitution between different types of capital.

## General Case: Technology-Specific Labor Efficiency

For completeness, we present the general case where labor efficiency depends on both productivity and emissions changes, with skill-specific adaptation costs:

\[
\text{Eff}_L(t) = \frac{1}{1 + \gamma_{labor} \int_{\underline{\theta}}^{\overline{\theta}} \sum_i \theta_t^i \left[\psi_A(\theta)(A_t^i - A_{t-1}^i)^2 + \psi_\eta(\theta)(\eta_t^i - \eta_{t-1}^i)^2\right] dF(\theta)}
\]

Where:
- $\psi_A(\theta)$ and $\psi_\eta(\theta)$ are skill-specific adjustment cost functions for productivity and emissions changes
- Higher skilled workers (higher $\theta$) have lower values of $\psi_A(\theta)$ and $\psi_\eta(\theta)$
- The integral cannot be separated from the technology parameters as skill levels affect adaptation to specific technologies
- Portfolio shares $\theta_t^i$ weight the importance of each technology's changes

This formulation:
- Captures adjustment costs from changes in both productivity and emissions intensity
- Maintains the skill-technology specific interaction
- Allows for different adaptation costs for productivity versus emissions changes
- Reduces to simpler cases when skill distribution or technology parameters are uniform

---

## Technology Choice Mechanism

The firm's technology choice problem incorporates:

1. **Productivity-Emissions Trade-off:**
   - Higher productivity may be associated with higher emissions intensity
   - Trade-off governed by correlation coefficient $\rho$
   - Firms optimize over the joint distribution $f(A_t,\eta_t)$

2. **Portfolio Diversification:**
   - Diminishing returns prevent concentration in single technology
   - Portfolio shares must sum to one: $\sum_i \theta_t^i = 1$
   - Concave production function ensures diversification

---

## Solving the Model

Given the complexity introduced by the portfolio choice and diminishing marginal returns, the model requires numerical methods for solution:

1. **Initialization:**
   - Set initial values for \( K_{init} \), \( A_{init} \), \( \eta_{init} \).
   - Specify the set of available technologies \( \{ A_t^i, \eta_t^i \} \).

2. **Optimization:**
   - Firms choose \( \theta_t^i \) to maximize profits, considering diminishing returns and adjustment costs.
   - Households choose \( C_t \) and \( L_t \) to maximize utility.

3. **Equilibrium Conditions:**
   - Markets must clear in each period.
   - Prices \( w_t \) and \( r_t \) adjust to equilibrate supply and demand.

4. **Numerical Methods:**
   - Use gradient-based optimization algorithms.
   - Implement constraints to ensure feasibility.

---

