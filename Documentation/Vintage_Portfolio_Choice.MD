# Using Portfolio Choice to Model Structural Compostion and Change

## Overview

The goal is to find the firm's optimal portfolio choice across different vintages while ensuring that the structural composition of the economy is maintained and allowing for structural change.

## Problem Formulation and Structure

### Variables and Parameters:

- Time Periods: $t \in \{0,1,...,T\}$
- Investment Amount: $V_{t,i}$ — investment in vintage $i$ at time $t$
- Productivity Factor: $A_{t,i}$ — base productivity of vintage $i$ at time $t$
- Emissions Intensity: $E_{t,i}$ — emissions intensity of vintage $i$ at time $t$
- Carbon Tax Rate: $\tau_t$ at time $t$
- Technological Progress Functions: $g(i,t)$ and $h(i,t)$ — functions representing expected shifts in productivity and emissions intensity for vintage $i$ due to technological progress at time $t$
- Discount Factor: $\beta$

### Core Functions

#### Productivity Function

We model the productivity from investing in vintage $i$ at time $t$ as a concave function:

$$\Pi_{t,i} = A_{t,i}V_{t,i}^\alpha, \text{ with } 0 < \alpha < 1$$

Where $\alpha$ represents the degree of diminishing returns.

#### Emissions Function

$$\Phi_{t,i} = E_{t,i}V_{t,i}$$

### Objective Function

The firm's goal is to maximize the discounted cumulative productivity net of emissions costs:

$$\max_{\{V_{t,i}\}} \sum_{t=0}^T \beta^t(\sum_i [A_{t,i}V_{t,i}^\alpha - \tau_t E_{t,i}V_{t,i}])$$

### Constraints

Budget Constraint:
$$\sum_i V_{t,i} \leq K, \forall t$$

Non-negativity of Investment:
$$V_{t,i} \geq 0, \forall i,t$$

Technological Progress Dynamics:
$$A_{t+1,i} = A_{t,i} + g(i,t)$$
$$E_{t+1,i} = E_{t,i} - h(i,t)$$

### Additional Details

- The firm is a representative agent investing across the entire economy, covering all sectors.
- Each vintage $i$ has its own technological progress functions $g(i,t)$ and $h(i,t)$.
- The firm's portfolio choice should not destroy the structural composition of the economy.
- The goal is to find sufficient conditions to prevent the firm from investing all capital into a single vintage.

## Solution Derivation

### Step 1: Setting Up the Lagrangian

We construct the Lagrangian $L$ for the optimization problem at each time $t$:

$$L = \sum_i [A_{t,i}V_{t,i}^\alpha - \tau_t E_{t,i}V_{t,i}] + \lambda_t(K - \sum_i V_{t,i})$$

Where $\lambda_t$ is the Lagrange multiplier associated with the budget constraint at time $t$.

### Step 2: First-Order Conditions

For each vintage $i$, the first-order condition (FOC) with respect to $V_{t,i}$ is:

$$\frac{\partial L}{\partial V_{t,i}} = 0$$

Calculating the derivative:

$$\frac{\partial L}{\partial V_{t,i}} = \alpha A_{t,i}V_{t,i}^{\alpha-1} - \tau_t E_{t,i} - \lambda_t = 0$$

Rewriting:

$$\alpha A_{t,i}V_{t,i}^{\alpha-1} = \tau_t E_{t,i} + \lambda_t$$

### Step 3: Solving for Optimal Investment $V_{t,i}^*$

We can solve for $V_{t,i}$:

$$V_{t,i}^* = (\frac{\tau_t E_{t,i} + \lambda_t}{\alpha A_{t,i}})^{\frac{1}{\alpha-1}}$$

Since $\alpha-1 < 0$ (because $0 < \alpha < 1$), the exponent $\frac{1}{\alpha-1}$ is negative, which maintains the concavity of the function.

### Step 4: Determining the Lagrange Multiplier $\lambda_t$

The Lagrange multiplier $\lambda_t$ adjusts to ensure the budget constraint:

$$\sum_i V_{t,i}^* = K$$

This gives us:

$$\sum_i (\frac{\tau_t E_{t,i} + \lambda_t}{\alpha A_{t,i}})^{\frac{1}{\alpha-1}} = K$$

#### Special Case: Homogeneous Vintages

When vintages are identical ($A_{t,i} = A_t, E_{t,i} = E_t, \forall i$), with $N$ vintages:

$$V_{t,i}^* = \frac{K}{N}$$

And $\lambda_t$ has closed form:

$$\lambda_t = \alpha A_t (\frac{K}{N})^{\alpha-1} - \tau_t E_t$$

## Law of Motion for Investment

Investment in each vintage evolves over time due to changes in $A_{t,i}$ and $E_{t,i}$:

$$V_{t+1,i}^* = (\frac{\tau_{t+1} E_{t+1,i} + \lambda_{t+1}}{\alpha A_{t+1,i}})^{\frac{1}{\alpha-1}}$$

Where:

$$A_{t+1,i} = A_{t,i} + g(i,t)$$
$$E_{t+1,i} = E_{t,i} - h(i,t)$$

### Implications:

- Technological Progress: Increases in $A_{t,i}$ due to $g(i,t)$ raise optimal investment $V_{t,i}^*$
- Emissions Reduction: Decreases in $E_{t,i}$ due to $h(i,t)$ raise $V_{t,i}^*$ by reducing the negative impact of the carbon tax
- Carbon Tax Dynamics: Changes in $\tau_t$ influence investment decisions, with higher taxes discouraging investment in high-emission vintages

## Policy Implications and Economic Interpretation

1. Investment Diversification:
   - The firm invests in all vintages but allocates more capital to those with higher net productivity after adjusting for emissions costs

2. Role of Technological Progress:
   - Vintages that experience faster technological progress ($g(i,t)$ increases) become more attractive for investment
   - Emissions-reducing technologies ($h(i,t)$ increases) also enhance a vintage's attractiveness by lowering emissions costs

3. Carbon Tax Impact:
   - A higher carbon tax $\tau_t$ discourages investment in high-emission vintages, shifting capital towards cleaner technologies
   - This aligns the firm's investment decisions with environmental policy objectives

4. Structural Change and Economic Evolution:
   - The economy evolves as investments shift toward vintages with better productivity-emissions profiles
   - Structural change occurs endogenously, guided by technological progress and policy incentives

## Conclusion

By integrating diminishing returns into the original optimization problem, we have developed a comprehensive model that:

- Prevents Over-Concentration: The concavity in the production function ensures that the firm does not invest all capital into a single vintage
- Allows Structural Change: The model captures how technological progress and emissions reductions influence investment decisions over time
- Provides Optimal Portfolio Choice: Derives the optimal investment allocation across vintages at each time $t$, accounting for productivity, emissions intensity, and the carbon tax
- Maintains Structural Composition: Ensures a diversified investment portfolio, preserving the economy's structural composition while allowing for evolution

## Final Remarks

- Model Flexibility: The framework can accommodate different functional forms for $g(i,t)$ and $h(i,t)$, as well as other extensions like uncertainty or sector-specific policies
- Applicability: The model provides valuable insights for policymakers on how carbon taxes and technological advancements can influence investment patterns and drive sustainable economic growth
- Further Analysis: Numerical simulations can be performed to explore specific scenarios, calibrate the model with real data, and assess the impact of different policy parameters