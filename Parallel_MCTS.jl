# Import necessary packages
using Random
using Statistics
using Distributions
using Base.Threads
using BenchmarkTools

include("model_parameters.jl")
using .ModelParametersModule

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
    labor_efficiency::Float64    # New field
    technology_split::Float64    # New field for tracking investment split
end

struct PolicyAction
    τ_current::Float64
    τ_announced::Float64
end

# Modify the MCTSNode struct to include an 'expanded' atomic flag and remove the lock
mutable struct MCTSNode
    state::PolicyState
    action::Union{Nothing, PolicyAction}
    parent::Union{Nothing, MCTSNode}
    children::Vector{MCTSNode}
    visits::Atomic{Int}
    total_value::Atomic{Float64}
    expanded::Atomic{Bool}  # New atomic flag to indicate expansion

    # Constructor for root node
    function MCTSNode(state::PolicyState)
        new(state, nothing, nothing, [],
            Atomic{Int}(0), Atomic{Float64}(0.0), Atomic{Bool}(false))
    end

    # Constructor for child nodes
    function MCTSNode(state::PolicyState, action::PolicyAction, parent::MCTSNode)
        new(state, action, parent, [],
            Atomic{Int}(0), Atomic{Float64}(0.0), Atomic{Bool}(false))
    end
end

# Optimized for production parallel processing
const BATCH_SIZE = 20  # Reduced from 200 to better match our iteration count

# Add this function before the MCTS core functions
function step_environment(state::PolicyState, action::PolicyAction, debug_print::Bool, params::ModelParameters)
    # Create tax expectations
    tax_expectations = form_tax_expectations(
        action.τ_current,
        action.τ_announced,
        state.technology_params["μ_η"],
        state.technology_params["σ_η"],
        state.credibility
    )
    
    # Get RANK equilibrium with params
    equilibrium = compute_equilibrium(tax_expectations, params)
    
    # Check for error in equilibrium computation
    if haskey(equilibrium, "error")
        if debug_print
            println("\nEquilibrium computation failed:")
            println(get(equilibrium, "error_message", "Unknown error"))
        end
        
        # Return penalized state with proper boolean error flag
        return PolicyState(
            state.time + 1,
            Dict{String, Float64}(
                "error_state" => true,
                "Y_t" => 0.0,
                "η_t" => state.economic_state["η_t"],
                "Labor_Efficiency" => 0.0,
                "Technology_Split" => 0.0
            ),
            state.emissions * 1.1,
            state.θ_mean,
            state.θ_std,
            vcat(state.tax_history, action.τ_current),
            state.credibility * 0.9,
            state.technology_params,
            0.0,
            0.0
        )
    end
    
    # Normal state update for successful computation
    new_labor_efficiency = get(equilibrium, "Labor_Efficiency", 1.0)
    new_emissions = state.emissions + equilibrium["η_t"] * equilibrium["Y_t"]
    θ_new = state.θ_mean
    θ_std_new = state.θ_std * 0.95
    
    return PolicyState(
        state.time + 1,
        equilibrium,
        new_emissions,
        θ_new,
        θ_std_new,
        vcat(state.tax_history, action.τ_current),
        state.credibility,
        state.technology_params,
        new_labor_efficiency,
        get(equilibrium, "Technology_Split", 0.5)
    )
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

# Remove locks in the select_node function
function select_node(node::MCTSNode, params::ModelParameters)
    current = node
    visit_count = 0
    max_visits = 1000  # Prevent infinite loops

    while !isempty(current.children) && visit_count < max_visits
        visit_count += 1

        unvisited = filter(child -> child.visits.value == 0, current.children)

        if !isempty(unvisited)
            chosen = first(unvisited)
            return chosen
        end

        # Select best child using UCB
        parent_visits = current.visits.value
        chosen = argmax(child -> ucb_score(child, parent_visits, params), current.children)
        current = chosen
    end

    return current
end

# Update the expand_node function to use the atomic 'expanded' flag
function expand_node(node::MCTSNode, params::ModelParameters)
    if atomic_cas!(node.expanded, false, true) == false
        # Node was not previously expanded; proceed to expand
        current_tax = isnothing(node.action) ? 0.0 : node.action.τ_current
        valid_actions = get_valid_actions(current_tax, params)

        for action in valid_actions
            next_state = step_environment(node.state, action, false, params)
            child = MCTSNode(next_state, action, node)
            push!(node.children, child)
        end
    else
        # Node was already expanded by another thread; do nothing
        return
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
    next_state = step_environment(node.state, action, false, params)
    
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

# Adjust the mcts_search function to reflect the lock-free operations
function mcts_search(root_state::PolicyState, n_iterations::Int, max_depth::Int, params::ModelParameters)
    root = MCTSNode(root_state)
    batch_size = min(params.batch_size, n_iterations)  # Control batch size

    for batch_start in 1:batch_size:n_iterations
        batch_end = min(batch_start + batch_size - 1, n_iterations)
        current_batch_size = batch_end - batch_start + 1

        results = Vector{Tuple{MCTSNode, Float64}}(undef, current_batch_size)

        # Parallel processing without locks
        @threads for i in 1:current_batch_size
            node = select_node(root, params)

            # Expand node if not already expanded
            if node.visits.value > 0 && isempty(node.children)
                expand_node(node, params)
                node = isempty(node.children) ? node : rand(node.children)
            end

            value = simulate(node, max_depth, params)
            results[i] = (node, value)
        end

        # Parallel backpropagation using atomic operations
        @threads for (node, value) in results
            backpropagate(node, value)
        end
    end

    # Choose the best action based on visit counts
    if isempty(root.children)
        @warn "No children expanded in root node"
        return PolicyAction(0.0, 0.0)
    end

    return argmax(child -> child.visits.value, root.children).action
end

# Evaluation Functions
function evaluate_state(state::PolicyState, params::ModelParameters)
    # Check if state is economically infeasible
    if haskey(state.economic_state, "error_state")
        return -1000.0  # Large penalty for infeasible states
    end

    # Normal evaluation for feasible states
    output = state.economic_state["Y_t"]
    damages = damage_function(state.emissions, state.θ_mean)
    
    current_tax = isempty(state.tax_history) ? 0.0 : last(state.tax_history)
    tax_revenue = current_tax * output
    
    efficiency_penalty = (1.0 - state.labor_efficiency) * output * 0.1
    split_penalty = abs(state.technology_split - 0.5) * output * 0.05
    
    return output - damages + params.tax_revenue_weight * tax_revenue - 
           efficiency_penalty - split_penalty
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
        "μ_η" => params.μ_η,
        "σ_A" => params.σ_A,
        "σ_η" => params.σ_η,
        "ρ" => params.ρ
    )
    
    initial_expectations = form_tax_expectations(
        initial_tax,
        announced_tax,
        params.μ_η,
        params.σ_η,
        credibility
    )
    
    initial_econ_state = compute_equilibrium(initial_expectations, params)
    
    # Get initial labor efficiency (will be 1.0 for initial state)
    initial_labor_efficiency = 1.0
    
    return PolicyState(
        0,
        initial_econ_state,
        0.0,
        params.θ_init_mean,
        params.θ_init_std,
        Float64[],
        credibility,
        initial_tech_params,
        initial_labor_efficiency,
        get(initial_econ_state, "Technology_Split", 0.5)  # Default to balanced split
    )
end

# Export main functions
export PolicyState, PolicyAction, mcts_search, initialize_mcts
