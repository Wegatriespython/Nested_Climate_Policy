# Simplified Vintage Capital Model Strategy

## Core Framework

### Given
- **Initial Technologies**: `tech_minus1` with (A, η) characteristics
- **New Technologies**: `tech0` and `tech1` for subsequent periods
- **Carbon Price**: τ for cost calculations
- **Operation Modes**: Generated combinations of old and new technology characteristics

### Objective
Generate feasible portfolios combining existing capital (operated with new technology) and new investments.

## Implementation Structure

### 1. Data Structures
```julia
TechnologySet
    A::Vector{Float64}      # Productivity values
    η::Vector{Float64}      # Emissions intensity values

OperatedCapital
    weights::Matrix{Float64}          # Weights over existing capital
    base_tech::TechnologySet          # Original characteristics
    operation_mode::TechnologySet     # Operation modes with newer tech

NewInvestment
    weights::Matrix{Float64}          # Weights over new technologies
    technologies::TechnologySet       # New technology characteristics

VintagePortfolio
    existing_capital::OperatedCapital
    new_investment::NewInvestment
```

### 2. Portfolio Generation Process
1. **Operation Mode Generation**:
   - Create composite technologies representing how old capital can be operated with new tech
   - Uses `LinRange` to generate mode factors between old and new characteristics

2. **Weight Generation**:
   - Uses Dirichlet sampling to generate feasible weight combinations
   - Ensures weights sum to 1 and are non-negative
   - Separate weights for operated capital and new investments

3. **Portfolio Construction**:
   - Splits total investment between existing and new capital
   - Normalizes weights across both categories
   - Creates VintagePortfolio instances combining both components

### 3. Characteristic Calculations
- **Effective Characteristics**: (A_eff, η_eff)
  ```julia
  A_eff = sum(existing_weights * operated_A) + sum(new_weights * new_A)
  η_eff = sum(existing_weights * operated_η) + sum(new_weights * new_η)
  ```
- **Cost Calculation**:
  ```julia
  cost = A_eff - τ * η_eff
  ```

## Key Features

### 1. Dirichlet Sampling
- Provides diverse, valid weight combinations
- Controlled by concentration parameter α
- Ensures proper weight normalization

### 2. Operation Modes
- Represents technology mixing between periods
- Linear combinations of old and new characteristics
- Captures operational flexibility

### 3. Portfolio Distribution
- Visualized in 3D space (A_eff, η_eff, cost)
- Separate distributions for each period
- Includes density estimation

## Limitations
- Linear combination assumption for operation modes
- Fixed number of operation modes per technology
- Simplified transition dynamics between periods

## Output Format
- Sequence matrix containing:
  - Period 0: (A_eff_0, η_eff_0, cost_0)
  - Period 1: (A_eff_1, η_eff_1, cost_1)
- Visualization through enhanced 3D scatter plots

## Relationship to Complete Model

### Key Simplifications

1. **Dimensionality Reduction**
   - Complete model: Tracks full K_{v→w,t} matrix across all vintages
   - Simplified model: Collapses to two periods with limited vintage combinations
   - Justification: Captures essential dynamics while remaining computationally tractable

2. **Reallocation Mechanism**
   - Complete model: Explicit reallocation matrix R_{v→w,t} with costs
   - Simplified model: Operation modes represent implicit reallocation through linear combinations
   - Justification: Avoids complex optimization while preserving technology mixing effects

3. **Price Formation**
   - Complete model: Endogenous price formation with no-arbitrage conditions
   - Simplified model: Direct cost calculation based on productivity and emissions
   - Justification: Focuses on physical characteristics rather than market dynamics

4. **State Space**
   - Complete model: Full dynamic evolution of capital stock
   - Simplified model: Static portfolio generation for two periods
   - Justification: Enables exploration of feasible combinations without path dependency

### Preserved Features

1. **Technology Characteristics**
   - Both models maintain core (A, η) characteristics
   - Productivity and emissions remain key decision variables
   - Environmental-economic trade-offs preserved

2. **Capital Heterogeneity**
   - Multiple technology vintages
   - Mixing between old and new technologies
   - Within-vintage variation through operation modes

3. **Portfolio Structure**
   - Weighted combinations of technologies
   - Conservation of total investment
   - Feasibility constraints on weights