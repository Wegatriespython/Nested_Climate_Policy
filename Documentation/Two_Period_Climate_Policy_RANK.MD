# Theoretical Formulation of a Two-Period RANK Model with Climate Policy Extensions

## Introduction
This document presents a theoretical formulation of a two-period Representative Agent New Keynesian (RANK) model with climate policy extensions. The model explicitly incorporates the joint distribution of capital productivity and carbon intensity, allowing firms to face trade-offs based on the correlation between these factors. Additionally, it introduces a distribution of worker skills inversely related to job adjustment costs, capturing labor market frictions associated with changes in firms' carbon intensity.

## Overview
The model examines the decisions of firms and households over two periods, integrating climate policy through an emission tax. Firms can continuously adjust their capital stock without adjustment costs, choosing capital varieties with different productivity and carbon intensity levels. Workers' adjustment costs in the labor market depend on both their skill levels and changes in the carbon intensity of firms' capital stock.

## Model Structure

### 1. Firms

#### Capital Choice and Production Technology
Capital Varieties: Firms choose capital types characterized by productivity $A_t$ and carbon intensity $\eta_t$.
Joint Distribution: The pair $(A_t,\eta_t)$ follows a joint distribution with means $\mu_A$, $\mu_\eta$, standard deviations $\sigma_A$, $\sigma_\eta$, and correlation coefficient $\rho$.

Probability Density Function:
$$f(A_t,\eta_t)=f_{A,\eta}(A_t,\eta_t)$$

Trade-off: Higher productivity $A_t$ may be associated with higher carbon intensity $\eta_t$, depending on $\rho$.

Production Function:
$$Y_t=A_tK_t^\alpha L_t^{1-\alpha}$$

$Y_t$: Output at time $t$
$K_t$: Capital stock
$L_t$: Labor input
$\alpha\in(0,1)$: Capital share parameter

Emission Tax and Profits
Emission Tax:
$$Tax_t=\tau\eta_tY_t$$
$\tau\geq0$: Emission tax rate

Profit Maximization Problem:
$$\max_{A_t,\eta_t,K_t,L_t}[Y_t-w_tL_t-r_tK_t-\tau\eta_tY_t]$$

Constraints:
Firms can freely adjust $K_t$ and choose $(A_t,\eta_t)$ without adjustment costs.
Choices of $(A_t,\eta_t)$ are constrained by the joint distribution $f_{A,\eta}(A_t,\eta_t)$.

First-Order Conditions
Optimal Choice of Capital and Labor:
Labor Demand:
$$\frac{\partial\Pi_t}{\partial L_t}=0\implies w_t=(1-\alpha)(1-\tau\eta_t)\frac{Y_t}{L_t}$$

Capital Demand:
$$\frac{\partial\Pi_t}{\partial K_t}=0\implies r_t=\alpha(1-\tau\eta_t)\frac{Y_t}{K_t}$$

Optimal Choice of $(A_t,\eta_t)$:
Firms select $(A_t,\eta_t)$ to maximize after-tax output per unit of capital and labor.
Given the joint distribution, firms face a trade-off between higher productivity and higher emissions taxes.

### 2. Households

#### Preferences and Utility
Utility Function:
$$U=\sum_{t=0}^1\beta^t[\frac{C_t^{1-\sigma}}{1-\sigma}-\chi\frac{L_t^{1+\nu}}{1+\nu}]-\text{Adjustment Cost}_t$$

$C_t$: Consumption
$L_t$: Labor supply
$\beta\in(0,1)$: Discount factor
$\sigma>0$: Relative risk aversion coefficient
$\chi>0$: Labor disutility weight
$\nu\geq0$: Inverse of Frisch elasticity

Adjustment Cost:
$$\text{Adjustment Cost}_t=\int_{\underline{\theta}}^{\overline{\theta}}\phi(\Delta\eta_t)\psi(\theta)dF(\theta)$$

$\Delta\eta_t=|\eta_t-\eta_{t-1}|$: Change in carbon intensity
$\theta\in[\underline{\theta},\overline{\theta}]$: Worker skill level
$F(\theta)$: Cumulative distribution function of skills
$\phi(\cdot)$: Adjustment cost function increasing in $\Delta\eta_t$
$\psi(\cdot)$: Function decreasing in $\theta$; higher skills imply lower costs

Budget Constraints
Period 0:
$$C_0+S_0=w_0L_0-\text{Adjustment Cost}_0$$
$S_0$: Savings (investment in capital for period 1)

Period 1:
$$C_1=w_1L_1+r_1S_0$$

Labor Supply Decision
First-Order Condition for Labor:
$$\chi L_t^\nu=C_t^{-\sigma}w_t$$

Consumption Euler Equation:
$$\beta(\frac{C_{t+1}}{C_t})^\sigma=1+r_{t+1}-\delta$$

### 3. Government

#### Budget Constraint
$$T_t=\tau\eta_tY_t$$
$T_t$: Lump-sum transfers rebated to households

### 4. Market Clearing Conditions

#### Goods Market
$$Y_t=C_t+K_{t+1}-(1-\delta)K_t+\text{Adjustment Cost}_t$$

#### Labor Market
$$L_t^{demand}=L_t^{supply}$$

#### Capital Market
$$K_t^{demand}=K_t^{supply}=S_0$$

## Expectations and Uncertainty
Stochastic Variables: 
$A_t$ and 
$\eta_t$ are stochastic and jointly distributed.
Joint Distribution:
Mean productivity: 
$\mu_A$ 
Mean carbon intensity: 
$\mu_\eta$ 
Standard deviations: 
$\sigma_A$, 
$\sigma_\eta$ 
Correlation coefficient: 
$\rho$
Agents' Expectations: Rational expectations formed based on 
$f_{A,\eta}(A_t,\eta_t)$.

### Model Solution
Firms' Problem
Given Prices: Take $w_t$ and $r_t$ as given.
Optimal $(A_t,\eta_t)$ Choice: Profit per Unit of Capital and Labor:
$$\Pi_t=[1-\tau\eta_t]A_tK_t^\alpha L_t^{1-\alpha}-w_tL_t-r_tK_t$$

#### Effective Productivity:
$$A_t^{eff}=(1-\tau\eta_t)A_t$$

#### Trade-off Analysis:
Firms evaluate $A_t^{eff}$ and decide on $(A_t,\eta_t)$ that maximizes profits.
The correlation $\rho$ influences the choice: Positive $\rho$ implies higher $A_t$ associated with higher $\eta_t$.

Optimal Input Demands:
Given $A_t^{eff}$, firms demand labor and capital according to the marginal productivity conditions.

Households' Problem
Labor Supply:
Adjust labor supply $L_t$ based on wages $w_t$ and adjustment costs.High adjustment costs (due to significant $\Delta\eta_t$ and low 
$\theta$) reduce net utility from working.

Consumption and Savings:
Decide on $C_t$ and $S_0$ using the Euler equation and budget constraints.
Aggregate Adjustment Cost:
$$\text{Adjustment Cost}_t=\gamma(\Delta\eta_t)^2\int_{\underline{\theta}}^{\overline{\theta}}\frac{1}{\theta}dF(\theta)$$

### Equilibrium Computation
Initialization: Guess initial prices $w_t$ and $r_t$.
Firms' Decision: For given prices, firms select $(A_t,\eta_t)$ to maximize profits.
Determine aggregate labor demand $L_t^{demand}$ and capital demand $K_t^{demand}$.
Households' Decision: Given wages and returns, households decide $C_t$, $L_t^{supply}$, and $S_0$. Compute total adjustment costs.
Market Clearing: Check if: $$L_t^{demand}=L_t^{supply}$$ $$K_t^{demand}=K_t^{supply}=S_0$$ Goods market clears.
Update Prices: Adjust $w_t$ and $r_t$ to clear markets.
Iteration: Repeat steps 2-5 until convergence.

### Computational Approach:
Use numerical methods to solve the model due to the non-linearity introduced by the joint distribution and adjustment costs.
Monte Carlo simulations can approximate expectations over 
$A_t$ and $\eta_t$.
Skill Distribution: Choose an appropriate distribution $F(\theta)$, e.g., uniform or log-normal.
Policy Analysis:
Investigate how changes in $\tau$, $\gamma$, and $\rho$ affect the equilibrium outcomes.
Assess the impact of emission taxes on firms' technology choices and labor market dynamics.
Extension to Heterogeneous Agents: 
While the current model is a RANK framework, extending it to a HANK model would involve introducing heterogeneity in agents' wealth and consumption.

## Conclusion
This theoretical formulation integrates a joint distribution of capital productivity and carbon intensity into a two-period RANK model with climate policy considerations. Firms face a trade-off between productivity and carbon emissions due to the emission tax and the correlation between $A_t$ and $\eta_t$. Workers experience adjustment costs related to changes in firms' carbon intensity, inversely proportional to their skill levels. The model provides a feasible framework to analyze the effects of climate policy on macroeconomic variables, firms' technology adoption, and labor market dynamics.
