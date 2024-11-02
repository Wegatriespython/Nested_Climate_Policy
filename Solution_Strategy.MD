# Technology Portfolio Optimization Strategy
## Overview of Solution Architecture

### 1. Theoretical Foundation: Field Equations
The system is fundamentally described by field equations capturing:
- Capital evolution across vintage-technology pairs
- Labor efficiency adaptation
- Portfolio transition dynamics
- Cost and productivity relationships

Key moments of the system:
```julia
struct SystemMoments
    A_eff::Float64    # Effective productivity
    η_eff::Float64    # Effective emissions intensity
    cost::Float64     # Economic cost (function of A_eff, η_eff)
end
```

### 2. Discrete Technology Space
#### 2.1 Base Structure
```julia
struct TechnologySet
    A::Vector{Float64}    # Discrete productivity points
    η::Vector{Float64}    # Discrete emissions points
end
```

#### 2.2 Portfolio Composition
- Weighted combinations of technologies
- Operation modes between technology pairs
- Split between existing and new capital

#### 2.3 Key Challenge
Even with continuous weights, the resulting space:
- Is bounded by convex hull of technology points
- Contains unreachable "holes"
- Has discrete jumps between feasible points

### 3. Domain Mapping Solution
#### 3.1 Core Concept
Transform between:
- Original Domain (A_Hole): Discontinuous but economically valid
- Mapped Domain (A_Fixed): Continuous and solver-friendly

#### 3.2 Mapping Implementation
```julia
# Forward mapping (normalized → technology space)
function map_to_tech_space(x::Float64, tech_points::Vector{Float64})
    # Smooth piecewise linear interpolation
    # Returns valid technology point
end

# Inverse mapping (technology → normalized space)
function map_from_tech_space(y::Float64, tech_points::Vector{Float64})
    # Inverse transformation
    # Returns point in [0,1]
end
```

#### 3.3 Properties
- Bijective: Each point has unique mapping
- Differentiable: Suitable for gradient-based optimization
- Preserves ordering: Maintains economic meaning
- Bounded: Maps to valid technology combinations

### 4. Period Dependencies
#### 4.1 Constraint Structure
```julia
# In normalized space
@constraint(model, x_1 ≥ x_0)  # Progress constraint

# Maps to technology space as:
A_eff_1 ≥ A_eff_0  # Productivity improves
η_eff_1 ≤ η_eff_0  # Emissions decrease
```

#### 4.2 Feasibility Preservation
- Period 0 choices constrain Period 1 domain
- Maintains technological irreversibility
- Simplifies path dependency

### 5. Integration with GE Solver
#### 5.1 Variable Definition
```julia
# In GE model
@variable(model, x_0[1:n_dims] in [0,1])  # Normalized space
@variable(model, x_1[1:n_dims] in [0,1])

# Mapped variables
A_eff_0 = map_to_tech_space(x_0[1], tech.A)
η_eff_0 = map_to_tech_space(x_0[2], tech.η)
```

#### 5.2 Economic Constraints
```julia
# Production function
Y_t = (1 - τ_t * η_eff_t) * A_eff_t * K_t^α * L_t^(1-α)

# Investment constraints
I_t ≥ 0
K_t+1 = (1-δ)K_t + I_t
```

### 6. Implementation Notes
#### 6.1 Numerical Considerations
- Use smooth transition functions (sigmoid)
- Control condition number through mapping design
- Handle edge cases in technology space

#### 6.2 Performance Optimization
```julia
# Pre-compute feasible sequences
sequences = generate_vintage_portfolios(tech_minus1, tech0)
sequence_matrix = portfolios_to_matrix(sequences)

# Cache mapped values
mapped_points = Dict{Float64, Float64}()
```

#### 6.3 Validation Checks
- Test bijectivity of mappings
- Verify gradient properties
- Ensure economic constraints satisfied
- Check for numerical stability

### 7. Extension Points
1. **Adaptive Precision**
   - Adjust mapping resolution based on solver progress
   - Finer grain near optimal points

2. **Multi-Period Extension**
   - Handle longer sequences of technology choices
   - Maintain computational tractability

3. **Uncertainty Integration**
   - Add stochastic elements to mapping
   - Preserve differentiability

### 8. Usage Example
```julia
# Define technology sets
tech_minus1 = TechnologySet([1.0, 1.2], [0.8, 0.6])
tech0 = TechnologySet([1.3, 1.5], [0.5, 0.4])

# Generate feasible portfolios
portfolios = generate_vintage_portfolios(tech_minus1, tech0)

# Create mappings
x_space = create_normalized_space(portfolios)
mapping = create_bijective_mapping(x_space, portfolios)

# Use in GE solver
model = create_ge_model(mapping)
optimize!(model)
```

This strategy effectively transforms a complex, discrete technology choice problem into a tractable optimization while maintaining economic validity and solver compatibility.