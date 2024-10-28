# Import necessary packages
using Random
using Statistics
using Distributions
using Base.Threads
using BenchmarkTools

# Import RANK model
include("model_parameters.jl")

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
    lock::ReentrantLock  # Changed to ReentrantLock for better deadlock prevention
    
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
const MCTS_PARAMS = (
    exploration_constant = 2.0,
    tax_changes = [-0.10, -0.05, 0.0, 0.05, 0.10],
    min_tax = 0.0,
    max_tax = 0.30,
    discount_factor = 0.96
)

# Optimized for production parallel processing
const BATCH_SIZE = 20  # Reduced from 200 to better match our iteration count

# Add this constant with other MCTS parameters
const EVALUATION_PARAMS = (
    tax_revenue_weight = 0.5,  # Weight for tax revenue in utility function
)

# Add this function before the MCTS core functions
function step_environment(state::PolicyState, action::PolicyAction, debug_print::Bool, params::ModelParameters)
    # Create PolicyExpectations for RANK model
    tax_expectations = form_tax_expectations(
        action.τ_current,
        action.τ_announced,
        state.technology_params["μ_eta"],
        state.technology_params["σ_eta"],
        state.credibility
    )
    
    try
        # Get RANK equilibrium with params
        equilibrium = compute_equilibrium(tax_expectations, params)
        
        # Update emissions
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
            println("\nError occurred with these parameters:")
            println("Technology params:")
            for (k,v) in state.technology_params
                println("  $k: $v")
            end
            println("Tax history: $(state.tax_history)")
            println("Credibility: $(state.credibility)")
        end
        rethrow(e)
    end
end

# Core MCTS Functions
function get_valid_actions(current_tax::Float64, params::ModelParameters)
    actions = PolicyAction[]
    
    # Ensure current_tax is non-negative
    current_tax = max(0.0, current_tax)
    
    for Δτ₁ in params.tax_changes
        τ₁ = max(0.0, current_tax + Δτ₁)  # Ensure τ₁ is non-negative
        
        if τ₁ ≤ params.max_tax  # Fixed Unicode issue here
            for Δτ₂ in params.tax_changes
                τ₂ = max(0.0, τ₁ + Δτ₂)  # Ensure τ₂ is non-negative
                
                if τ₂ ≤ params.max_tax
                    # Only add action if both taxes are non-negative
                    push!(actions, PolicyAction(τ₁, τ₂))
                end
            end
        end
    end
    
    # If no valid actions found (shouldn't happen with our bounds), provide safe default
    if isempty(actions)
        push!(actions, PolicyAction(current_tax, current_tax))
    end
    
    return actions
end

function ucb_score(node::MCTSNode, parent_visits::Int, params::ModelParameters)
    visits = node.visits.value  # Access atomic value directly
    if visits == 0
        return Inf
    end
    
    total = node.total_value.value  # Access atomic value directly
    exploitation = total / visits
    exploration = params.exploration_constant * sqrt(log(parent_visits) / visits)
    
    return exploitation + exploration
end

function select_node(node::MCTSNode, params::ModelParameters)
    current = node
    visit_count = 0
    max_visits = 1000  # Prevent infinite loops
    
    while !isempty(current.children) && visit_count < max_visits
        visit_count += 1
        
        # Try to acquire lock with timeout using trylock
        acquired = false
        for _ in 1:100  # Try for 100 times with small sleep
            if trylock(current.lock)
                acquired = true
                break
            end
            sleep(0.01)  # Sleep for 10ms between attempts
        end
        
        if !acquired
            @warn "Lock acquisition timeout during selection"
            return current
        end
        
        try
            unvisited = filter(child -> child.visits.value == 0, current.children)
            
            if !isempty(unvisited)
                chosen = first(unvisited)
                return chosen
            end
            
            # Select best child using UCB
            parent_visits = current.visits.value
            chosen = argmax(child -> ucb_score(child, parent_visits, params), current.children)
            current = chosen
        finally
            unlock(current.lock)
        end
    end
    
    return current
end

function expand_node(node::MCTSNode, params::ModelParameters)
    lock(node.lock) do
        if !isempty(node.children)
            return  # Already expanded
        end
        
        current_tax = isnothing(node.action) ? 0.0 : node.action.τ_current
        valid_actions = get_valid_actions(current_tax, params)
        
        for action in valid_actions
            next_state = step_environment(node.state, action, false, params)
            child = MCTSNode(next_state, action, node)
            push!(node.children, child)
        end
    end
end

function simulate(node::MCTSNode, max_depth::Int, params::ModelParameters)
    if max_depth == 0
        return evaluate_state(node.state, params)
    end
    
    current_tax = isnothing(node.action) ? 0.0 : node.action.τ_current
    valid_actions = get_valid_actions(current_tax, params)
    
    if isempty(valid_actions)
        return evaluate_state(node.state, params)
    end
    
    action = rand(valid_actions)
    next_state = step_environment(node.state, action, false, params)  # Debug prints off
    
    return evaluate_state(node.state, params) + 
           params.discount_factor * simulate(MCTSNode(next_state), max_depth - 1, params)
end

function backpropagate(node::MCTSNode, value::Float64)
    current = node
    while !isnothing(current)
        atomic_add!(current.visits, 1)
        atomic_add!(current.total_value, value)
        current = current.parent
    end
end

# Streamlined parallel MCTS implementation
function mcts_search(root_state::PolicyState, n_iterations::Int, max_depth::Int, params::ModelParameters)
    root = MCTSNode(root_state)
    batch_size = min(params.batch_size, n_iterations)  # Smaller batch size for better control
    
    for batch_start in 1:batch_size:n_iterations
        batch_end = min(batch_start + batch_size - 1, n_iterations)
        current_batch_size = batch_end - batch_start + 1
        
        results = Vector{Tuple{MCTSNode, Float64}}(undef, current_batch_size)
        
        # Parallel batch processing with timeout
        @threads for i in 1:current_batch_size
            node = select_node(root, params)
            
            # Single lock for expansion and selection
            lock(node.lock) do
                if node.visits.value > 0 && isempty(node.children)
                    expand_node(node, params)
                    node = isempty(node.children) ? node : rand(node.children)
                end
            end
            
            value = simulate(node, max_depth, params)
            results[i] = (node, value)
        end
        
        # Sequential backpropagation to prevent race conditions
        for (node, value) in results
            backpropagate(node, value)
        end
    end
    
    # Select best action based on visit count instead of value
    lock(root.lock) do
        if isempty(root.children)
            @warn "No children expanded in root node"
            return PolicyAction(0.0, 0.0)
        end
        
        return argmax(child -> child.visits.value, root.children).action
    end
end

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

# Production initialization function
function initialize_mcts(;
    initial_tax::Float64 = 0.0,
    announced_tax::Float64 = 0.0,
    credibility::Float64 = 0.8,
    params::ModelParameters = DEFAULT_PARAMS
)
    initial_tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => params.μ_eta,
        "σ_A" => params.σ_A,
        "σ_eta" => params.σ_eta,
        "ρ" => params.ρ
    )
    
    initial_expectations = form_tax_expectations(
        initial_tax,
        announced_tax,
        params.μ_eta,
        params.σ_eta,
        credibility
    )
    initial_econ_state = compute_equilibrium(initial_expectations, params)
    
    return PolicyState(
        0,
        initial_econ_state,
        0.0,
        params.θ_init_mean,
        params.θ_init_std,
        Float64[],
        credibility,
        initial_tech_params
    )
end

# Export main functions
export PolicyState, PolicyAction, mcts_search, initialize_mcts
