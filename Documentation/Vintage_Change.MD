### Complete General Vintage Capital Model with No-Arbitrage

**1. Fundamental Sets**:
- Time: $t \in T$
- Vintages: $v, w \in V(t)$  
  ($V(t)$ is the set of all vintages available at time $t$)

**2. State Variables**:
For each time $t$:
- Capital Matrix: $K_{v\rightarrow w,t}$ 
  (capital from vintage $v$ operating with vintage $w$ technology)
- Investment Vector: $I_{v,t}$ 
  (new capital invested in vintage $v$)
- Reallocation Matrix: $R_{v\rightarrow w,t}$ 
  (capital being reallocated from $v$ to $w$ technology)
- Depreciation Matrix: $\delta_{v\rightarrow w,t}$ 
  (depreciation rate depends on both origin vintage and operating technology)
- Reallocation Costs: $C_{v\rightarrow w,t} = P_{v,t} - P_{w,t} + \tau_{v\rightarrow w,t}$
  - Price differential $(P_{v,t} - P_{w,t})$ enforces no-arbitrage condition
  - where $\tau_{v\rightarrow w,t}$ represents physical modification and adjustment costs
  - Combined cost prevents "free" productivity gains from reallocation

**2a. Technology Compatibility**:
- Vintages v,w are compatible if physical/technical constraints allow reallocation
- Rv→w,t = 0 for incompatible pairs, where:
  - Technical limitations prevent retrofitting
  - Regulatory restrictions exist
  - Modification costs exceed economic viability

**3. Derived Variables**:
- Total Capital: $K_t = \sum_v \sum_w K_{v\rightarrow w,t}$
- Vintage Capital: $K_{v,t} = \sum_w K_{v\rightarrow w,t}$
- Technology Capital: $\hat{K}_{w,t} = \sum_v K_{v\rightarrow w,t}$
- Vintage Fractions: $frac_{v,t} = K_{v,t}/K_t$
- Technology Fractions: $\hat{frac}_{w,t} = \hat{K}_{w,t}/K_t$
- Allocation Fractions: $\alpha_{v\rightarrow w,t} = K_{v\rightarrow w,t}/K_t$

**4. Dynamic Evolution**:
For each $v,w$ pair:
$\Delta K_{v\rightarrow w,t} = I_{v,t}\cdot\mathbb{1}_{\{v=w\}} + R_{v\rightarrow w,t} - \delta_{v\rightarrow w,t}\cdot K_{v\rightarrow w,t}$

Where:
- $\mathbb{1}_{\{v=w\}}$ is indicator function (1 if $v=w$, 0 otherwise)
- New investment only enters diagonal elements

Market Price Evolution:
$P_{v,t} = \mathbb{E}_t[\sum_{s\geq t} \beta^{s-t}(\max\{A_{v\rightarrow w,s} - C_{v\rightarrow w,s} - E_{v\rightarrow w,s}\cdot\pi_s\})]$
where:
- $\pi_s$ is the emissions price
- Price reflects best possible use of capital across compatible technologies
- Maximum taken over all feasible reallocations w
- Includes both direct operation and reallocation possibilities

**5. Conservation Laws**:
- Capital Conservation: $\sum_v \sum_w K_{v\rightarrow w,t} = K_t$
- Flow Conservation: $\sum_v \sum_w R_{v\rightarrow w,t} = 0$
- Fraction Conservation: $\sum_v frac_{v,t} = \sum_w \hat{frac}_{w,t} = \sum_v \sum_w \alpha_{v\rightarrow w,t} = 1$

**6. No-Arbitrage Conditions**:
- Return Equivalence: $r_{v,t} = r_{w,t}$ for all $v,w$
  where $r_{v,t} = (A_{v\rightarrow v,t} + \Delta P_{v,t})/P_{v,t} - \delta_{v,t}$
- Price Consistency: $P_{v,t}$ reflects NPV of future returns

**7. Performance Characteristics**:
For each vintage-technology pair $(v,w)$:
- Productivity: $A_{v\rightarrow w,t}$
- Emissivity: $E_{v\rightarrow w,t}$
- Cost: $C_{v\rightarrow w,t}$
- Other Characteristics: $X_{v\rightarrow w,t}$

**8. Aggregate Performance Metrics**:
- Total Productivity: $A_t = \sum_v \sum_w \alpha_{v\rightarrow w,t}\cdot A_{v\rightarrow w,t}$
- Total Emissivity: $E_t = \sum_v \sum_w \alpha_{v\rightarrow w,t}\cdot E_{v\rightarrow w,t}$
- Total Cost: $C_t = \sum_v \sum_w \alpha_{v\rightarrow w,t}\cdot C_{v\rightarrow w,t}$

**9. Transition Functions**:
$S_{t+1} = T(S_t, I_t, R_t, \delta_t)$

Where state $S_t$ includes:
- Capital Matrix: $\{K_{v\rightarrow w,t}\}$
- Performance Matrices: $\{A_{v\rightarrow w,t}, E_{v\rightarrow w,t}, C_{v\rightarrow w,t}, X_{v\rightarrow w,t}\}$
- Control Variables: $\{I_t, R_t\}$
- System Parameters: $\{\delta_t\}$

**10. Constraints**:
- Non-negativity: $K_{v\rightarrow w,t}, I_{v,t}, R_{v\rightarrow w,t} \geq 0$
- Feasibility: $R_{v\rightarrow w,t} \leq K_{v\rightarrow v,t}$
- Investment Budget: $\sum_v I_{v,t} \leq B_t$
- Technology Compatibility: $R_{v\rightarrow w,t} = 0$ for incompatible $(v,w)$

**11. Optimization Objectives**:
May include combinations of:
- Maximize: $\sum_t \beta^t\cdot A_t$ (productivity)
- Minimize: $\sum_t \beta^t\cdot E_t$ (emissions)
- Minimize: $\sum_t \beta^t\cdot C_t$ (costs)
- Subject to: All constraints and dynamics

**11. Economic Balance**:
For any reallocation $R_{v\rightarrow w,t}$:
$NPV(R_{v\rightarrow w,t}) = \sum_{s\geq t} \beta^{s-t}(A_{w\rightarrow w,s} - A_{v\rightarrow v,s}) - C_{v\rightarrow w,t} = 0$

### Model Context and Key Assumptions

**1. Theoretical Foundation**
- Derived from a Representative Agent General Equilibrium (RAGE) framework
- The firm serves as a representative sample of economy-wide capital dynamics
- Primary focus on tracking both economic productivity and environmental impact at macro level

**2. Capital Conceptualization**
- Hybrid approach to capital modeling:
  - Physical characteristics (like industrial machinery)
  - Abstracted from liquidity constraints
  - Each unit characterized by productivity (A) and emissions (E) parameters
- Simplification allows tracking of both economic and environmental performance while maintaining tractability

**3. Technology and Vintage Dynamics**
- Non-monotonic technology progression possible
- Rational holding of older vintages due to:
  - Uncertainty in future carbon prices
  - Potential technology downswings
  - Strategic hedging against future uncertainty
- Technology change treated as exogenous to maintain model simplicity

**4. Intentional Abstractions**
- Price formation mechanism simplified to prevent arbitrage
- Liquidity constraints omitted
- Technological change treated as exogenous
- Carbon prices treated as exogenous but uncertain

**5. Model Purpose**
- Primary goal: Track economy-wide capital evolution and its environmental impact
- Balance between theoretical rigor and practical tractability
- Focus on representative behavior rather than individual firm optimization

**6. Capital Reallocation Mechanics**
- Total reallocation cost includes:
  - Economic value difference (price differential)
  - Physical modification costs (τv→w,t)
- This structure ensures:
  - No arbitrage opportunities exist
  - Prices reflect true marginal productivity differences
  - Reallocation decisions are economically justified

**7. Depreciation Structure**
- Depreciation rates (δv→w,t) depend on:
  - Original vintage characteristics (v)
  - Operating technology requirements (w)
  - Usage patterns and maintenance needs
- This dual dependence captures how different technology combinations affect capital longevity

**8. Price Formation and Returns**
- Prices reflect full spectrum of potential returns:
  - Direct operation value
  - Reallocation options
  - Future technology adaptation potential
- No-arbitrage ensures consistent valuation across all feasible uses