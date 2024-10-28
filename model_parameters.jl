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
    
    # Joint distribution parameters
    μ_A::Float64 = 1.0         # Mean productivity
    μ_eta::Float64 = 1.0       # Mean carbon intensity
    σ_A::Float64 = 0.2         # Std dev of productivity
    σ_eta::Float64 = 0.2       # Std dev of carbon intensity
    ρ::Float64 = 0.5           # Correlation coefficient
    
    # Skill distribution parameters
    θ_min::Float64 = 0.1       # Minimum skill level
    θ_max::Float64 = 1.0       # Maximum skill level
    n_agents::Int = 1000       # Number of agents/workers
    
    # MCTS parameters
    exploration_constant::Float64 = 2.0
    tax_changes::Vector{Float64} = [-0.10, -0.05, 0.0, 0.05, 0.10]
    min_tax::Float64 = 0.0
    max_tax::Float64 = 0.30
    discount_factor::Float64 = 0.96
    batch_size::Int = 20
    tax_revenue_weight::Float64 = 0.5
    
    # Damage function parameters
    θ_init_mean::Float64 = 0.001
    θ_init_std::Float64 = 0.0005
end

# Create the default parameters
const DEFAULT_PARAMS = ModelParameters()

# Export both the type and the constant
export ModelParameters, DEFAULT_PARAMS

end # module
