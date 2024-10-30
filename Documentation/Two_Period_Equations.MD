# Two Period Model Equations (Revised Capital Markets)

## Variables
- Consumption: $C_0$, $C_1$
- Labor Supply: $L_0$, $L_1$
- Investment: $K_0$, $K_1$
- Capital Stock: $K_{stock,0}$, $K_{stock,1}$
- Capital Replacement Fraction: $frac_1$
- Technology Levels: $A_{eff,0}$, $A_{eff,1}$
- Emissions Intensity: $\eta_{eff,0}$, $\eta_{eff,1}$
- Interest Rates: $r_0$, $r_1$
- Wages: $w_0$, $w_1$
- Adjustment Costs and Labor Efficiency: $\Delta\eta_0$, $\Delta\eta_1$, $Eff_L(0)$, $Eff_L(1)$
- $p_t^i$: Price of capital with technology type $i$ at time $t$
- $K_t^i$: Stock of capital with technology type $i$ at time $t$
- $x_t^{i,j}$: Volume of capital converted from type $i$ to type $j$ at time $t$
- $I_t$: Net investment (change in total capital stock) at time $t$

Where technology types are:
- init: Initial technology $(A_{init}, \eta_{init})$
- 0: Current period technology $(A_0, \eta_0)$
- 1: Next period technology $(A_1, \eta_1)$

## Parameters
- $\alpha$: Capital share in production
- $\beta$: Discount factor
- $\sigma$: Relative risk aversion
- $\delta$: Depreciation rate
- $\chi$: Disutility of labor parameter
- $\nu$: Inverse of labor supply elasticity
- $\gamma$: Adjustment cost parameter
- $\tau_0$, $\tau_1$: Carbon taxes
- $skill\_factor$: Affects adjustment costs

## Key Assumptions
1. Perfect capital markets with normalized prices (p_t^i = 1 for all i,t)
2. No terminal value for capital after period 1
3. Deterministic setting (no uncertainty)

## Equations

### 1. Capital Stock Evolution by Type

Period 0:
$$K_0^{init} = K_{init} - x_0^{init,0}$$
$$K_0^0 = x_0^{init,0} + I_0$$
$$K_{stock,0} = K_0^{init} + K_0^0$$

Period 1:
$$K_1^{init} = (1-\delta)K_0^{init} - x_1^{init,1}$$
$$K_1^0 = (1-\delta)K_0^0 - x_1^{0,1}$$
$$K_1^1 = x_1^{init,1} + x_1^{0,1} + I_1$$
$$K_{stock,1} = K_1^{init} + K_1^0 + K_1^1$$

### 2. Capital Market Conditions

Physical Capital Market Clearing:
$$\sum_i K_t^i = K_{stock,t}$$

Capital Conversion Feasibility:
$$x_t^{i,j} \leq K_t^i$$

Conversion Cost for Each Technology Switch:
$$conversion\_cost_t^{i,j} = \gamma||\eta^j - \eta^i||^2 \times skill\_factor \times Y_t$$

Total Conversion Expenditure:
$$total\_conversion\_cost_t = \sum_{i,j} conversion\_cost_t^{i,j} \times x_t^{i,j}$$

### 3. Effective Technology and Emissions Intensity

Period 0:
$$A_{eff,0} = \frac{K_0^{init}A_{init} + K_0^0A_0}{K_{stock,0}}$$
$$\eta_{eff,0} = \frac{K_0^{init}\eta_{init} + K_0^0\eta_0}{K_{stock,0}}$$

Period 1:
$$A_{eff,1} = \frac{K_1^{init}A_{init} + K_1^0A_0 + K_1^1A_1}{K_{stock,1}}$$
$$\eta_{eff,1} = \frac{K_1^{init}\eta_{init} + K_1^0\eta_0 + K_1^1\eta_1}{K_{stock,1}}$$

### 4. Capital Returns

Marginal Product of Capital (for each type i):
$$MPK_t^i = (1-\tau_t\eta_{eff,t})A_{eff,t}\alpha(K_t^i)^{\alpha-1}(Eff_L(t)L_t)^{1-\alpha}$$

Terminal Condition:
$$MPK_1^i = r_1 + \delta$$

### 5. Budget Constraints

Period 0:
$$C_0 + I_0 + total\_conversion\_cost_0 = Y_0$$

Period 1:
$$C_1 + I_1 + total\_conversion\_cost_1 = Y_1$$

### 6. Production Function
$$Y_t = (1-\tau_t\eta_{eff,t})A_{eff,t}K_{stock,t}^\alpha(Eff_L(t)L_t)^{1-\alpha}$$

### 7. Factor Prices
Interest Rate:
$$r_t = \alpha\frac{Y_t}{K_{stock,t}} - \delta$$

Wage Rate:
$$w_t = (1-\alpha)\frac{Y_t}{Eff_L(t)L_t}$$

### 8. Euler Equation
$$\sigma(\ln C_1 - \ln C_0) = \ln\beta + \ln(1+r_1)$$

### 9. Labor Supply First-Order Conditions
$$\ln\chi + \nu\ln L_t = \ln w_t - \sigma\ln C_t$$

### 10. Objective Function
$$U = \frac{C_0^{1-\sigma}}{1-\sigma} - \chi\frac{L_0^{1+\nu}}{1+\nu} + \beta(\frac{C_1^{1-\sigma}}{1-\sigma} - \chi\frac{L_1^{1+\nu}}{1+\nu})$$

### Additional Constraints

Non-negativity:
$$K_t^i \geq 0, x_t^{i,j} \geq 0$$

Investment Bounds:
$$I_t \geq -\delta K_{stock,t}$$
