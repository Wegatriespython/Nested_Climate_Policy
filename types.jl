# Import necessary packages
using Distributions
using Random

# Import the ModelParametersModule and use its exports explicitly
include("model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

# Core Types
struct PolicyExpectations
    τ_current::Float64      # Current tax rate τₜ
    τ_announced::Float64    # Announced future tax rate τₜ₊₁
    η_mean::Float64        # Mean technology level
    η_std::Float64        # Technology dispersion
    credibility::Float64    # Policy maker credibility
end

mutable struct State
    time::Int
    economic_state::Dict{String, Float64}  # Y_t, C_t, etc.
    emissions::Float64                     # E_t
    tax_history::Vector{Float64}          # History of tax rates
    technology_params::Dict{String, Float64}  # Add technology parameters
end

# Global constants
const N_SIMULATIONS = 1  # Number of simulations for Monte Carlo sampling
const DEBUG_PRINTS = false  # Control debug print statements

# Define the skill distribution F(θ) using ModelParametersModule
const θ_values = rand(
    Uniform(DEFAULT_PARAMS.θ_min, DEFAULT_PARAMS.θ_max), 
    DEFAULT_PARAMS.n_agents
)

# Distribution Setup
function setup_distributions(params::ModelParameters = DEFAULT_PARAMS)
    cov = params.ρ * params.σ_A * params.σ_eta
    Σ = [params.σ_A^2 cov; cov params.σ_eta^2]
    return MvNormal([params.μ_A, params.μ_eta], Σ)
end

const joint_dist = setup_distributions()

export PolicyExpectations, State, θ_values, N_SIMULATIONS, DEBUG_PRINTS, setup_distributions, joint_dist
