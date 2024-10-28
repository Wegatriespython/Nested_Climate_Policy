module ModelParametersModule

using Random

# Model Parameters Structure with default values
Base.@kwdef struct ModelParameters
    # Economic parameters
    β::Float64 = 0.96          # Discount factor
    σ::Float64 = 2.0           # Relative risk aversion coefficient
    χ::Float64 = 1.0           # Labor disutility weight
    ν::Float64 = 1.0           # Inverse of Frisch elasticity
    α::Float64 = 0.33          # Capital share parameter
    δ::Float64 = 0.1           # Depreciation rate
    γ::Float64 = 0.1           # Adjustment cost coefficient
    K_init::Float64 = 1.0       # Initial capital stock
    n_agents::Int = 1000        # Number of agents/workers
    
    # Technology distribution parameters
    μ_A::Float64 = 1.0         # Mean productivity
    μ_eta::Float64 = 1.0       # Mean carbon intensity
    σ_A::Float64 = 0.2         # Std dev of productivity
    σ_eta::Float64 = 0.2       # Std dev of carbon intensity
    ρ::Float64 = 0.5           # Correlation coefficient
    
    # Skill distribution parameters
    θ_min::Float64 = 0.1       # Minimum skill level
    θ_max::Float64 = 1.0       # Maximum skill level
end

# Create the default parameters
const DEFAULT_PARAMS = ModelParameters()

# Export both the type and the constant
export ModelParameters, DEFAULT_PARAMS

end # module
