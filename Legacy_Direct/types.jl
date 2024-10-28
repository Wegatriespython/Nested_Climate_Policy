# Import necessary packages
using Distributions
using Random

# Import the ModelParametersModule and use its exports explicitly
include("model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

"""
Represents policy expectations and credibility
"""
struct PolicyExpectations
    τ_current::Float64      # Current tax rate
    τ_announced::Float64    # Announced future tax rate
    η_mean::Float64        # Mean technology level
    η_std::Float64         # Technology dispersion
    credibility::Float64    # Policy maker credibility
end

"""
Represents the economic state
"""
struct State
    time::Int
    economic_state::Dict{String, Float64}
    emissions::Float64
    tax_history::Vector{Float64}
    technology_params::Dict{String, Float64}
end

# Remove global constants and make them function-based
function get_simulation_params()
    return (
        n_simulations = 100,
        debug_prints = true
    )
end

function get_skill_distribution(params::ModelParameters = DEFAULT_PARAMS)
    return rand(
        Uniform(params.θ_min, params.θ_max), 
        params.n_agents
    )
end

function setup_technology_distribution(params::ModelParameters = DEFAULT_PARAMS)
    cov = params.ρ * params.σ_A * params.σ_eta
    Σ = [params.σ_A^2 cov; cov params.σ_eta^2]
    return MvNormal([params.μ_A, params.μ_eta], Σ)
end

export PolicyExpectations, State, ModelParameters, DEFAULT_PARAMS,
       get_simulation_params, get_skill_distribution, setup_technology_distribution
