# Import necessary packages
using Random
using Statistics
using Distributions
using Base.Threads
using BenchmarkTools

# Import RANK model
include("model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

# Define MCTS-specific structures
struct PolicyState
    time::Int
    economic_state::Dict{String, Float64}
    emissions::Float64
    θ_mean::Float64    # Damage parameter mean
    θ_std::Float64     # Damage parameter std
    tax_history::Vector{Float64}
    credibility::Float64
    technology_params::Dict{String, Float64}
end

struct PolicyAction
    τ_current::Float64
    τ_announced::Float64
end

mutable struct MCTSNode
    state::PolicyState
    action::Union{Nothing, PolicyAction}
    parent::Union{Nothing, MCTSNode}
    children::Vector{MCTSNode}
    visits::Atomic{Int}
    total_value::Atomic{Float64}
    lock::ReentrantLock
    
    # Constructor for root node
    function MCTSNode(state::PolicyState)
        new(state, nothing, nothing, [], 
            Atomic{Int}(0), Atomic{Float64}(0.0), ReentrantLock())
    end
    
    # Constructor for child nodes
    function MCTSNode(state::PolicyState, action::PolicyAction, parent::MCTSNode)
        new(state, action, parent, [], 
            Atomic{Int}(0), Atomic{Float64}(0.0), ReentrantLock())
    end
end

# MCTS Parameters
const EVALUATION_PARAMS = (
    tax_revenue_weight = 0.5,  # Weight for tax revenue in utility function
)

# Evaluation Functions
function evaluate_state(state::PolicyState, params::ModelParameters)
    # Economic output
    output = state.economic_state["Y_t"]
    
    # Climate damages
    damages = damage_function(state.emissions, state.θ_mean)
    
    # Tax revenue (τ * Y)
    current_tax = isempty(state.tax_history) ? 0.0 : last(state.tax_history)
    tax_revenue = current_tax * output
    
    # Total utility: output - damages + weighted tax revenue
    return output - damages + params.tax_revenue_weight * tax_revenue
end

function damage_function(emissions::Float64, θ::Float64)
    # Quadratic damage function
    return θ * emissions^2
end

# Step environment function (modified version)
function step_environment(state::PolicyState, action::PolicyAction, debug_print::Bool, params::ModelParameters)
    # Create tax expectations
    tax_expectations = form_tax_expectations(
        action.τ_current,
        action.τ_announced,
        state.technology_params["μ_η"],
        state.technology_params["σ_η"],
        state.credibility
    )
    
    try
        # Use our new equilibrium solver
        equilibrium = compute_equilibrium(tax_expectations, params)
        
        # Update emissions using current period values
        new_emissions = state.emissions + equilibrium["η_t"] * equilibrium["Y_t"]
        
        # Update damage beliefs
        θ_new = state.θ_mean
        θ_std_new = state.θ_std * 0.95  # Simple uncertainty reduction
        
        # Return new state
        return PolicyState(
            state.time + 1,
            equilibrium,
            new_emissions,
            θ_new,
            θ_std_new,
            vcat(state.tax_history, action.τ_current),
            state.credibility,
            state.technology_params
        )
    catch e
        if debug_print
            println("Error in step_environment:")
            println(e)
        end
        rethrow(e)
    end
end

# Modify get_valid_actions to use params
function get_valid_actions(current_tax::Float64, params::ModelParameters)
    actions = PolicyAction[]
    
    for Δτ₁ in params.tax_changes
        τ₁ = max(params.min_tax, min(params.max_tax, current_tax + Δτ₁))
        
        for Δτ₂ in params.tax_changes
            τ₂ = max(params.min_tax, min(params.max_tax, τ₁ + Δτ₂))
            push!(actions, PolicyAction(τ₁, τ₂))
        end
    end
    
    return actions
end

# Add this function after the other function definitions
function initialize_mcts(;
    initial_tax::Float64 = 0.0,
    announced_tax::Float64 = 0.0,
    credibility::Float64 = 0.8,
    params::ModelParameters = DEFAULT_PARAMS
)
    # Create initial technology parameters
    initial_tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_η" => params.μ_η,
        "σ_A" => params.σ_A,
        "σ_η" => params.σ_η,
        "ρ" => params.ρ
    )
    
    # Get initial equilibrium
    initial_expectations = form_tax_expectations(
        initial_tax,
        announced_tax,
        params.μ_η,
        params.σ_η,
        credibility
    )
    
    initial_econ_state = compute_equilibrium(initial_expectations, params)
    
    # Create initial state
    return PolicyState(
        0,                  # time
        initial_econ_state, # economic state
        0.0,               # emissions
        params.θ_init_mean, # damage parameter mean
        params.θ_init_std,  # damage parameter std
        Float64[],         # tax history
        credibility,       # credibility
        initial_tech_params # technology parameters
    )
end

# Make sure it's exported
export PolicyState, PolicyAction, MCTSNode, step_environment, evaluate_state
export mcts_search, initialize_mcts, get_valid_actions  # Added initialize_mcts and get_valid_actions
