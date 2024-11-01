# Import necessary packages
using Random
using Statistics
using Distributions
using Base.Threads
using BenchmarkTools

# Import RANK model
include("State_function_RANK.jl")

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
    visits::Int
    total_value::Float64
    
    # Constructor for root node
    function MCTSNode(state::PolicyState)
        new(state, nothing, nothing, [], 0, 0.0)
    end
    
    # Constructor for child nodes
    function MCTSNode(state::PolicyState, action::PolicyAction, parent::MCTSNode)
        new(state, action, parent, [], 0, 0.0)
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

# At the top with other parameters
const BATCH_SIZE = 100  # Make this constant again

# Core MCTS Functions
function get_valid_actions(current_tax::Float64)
    actions = PolicyAction[]
    
    # Ensure current_tax is non-negative
    current_tax = max(0.0, current_tax)
    
    for Δτ₁ in MCTS_PARAMS.tax_changes
        τ₁ = max(0.0, current_tax + Δτ₁)  # Ensure τ₁ is non-negative
        
        if τ₁ ≤ MCTS_PARAMS.max_tax
            for Δτ₂ in MCTS_PARAMS.tax_changes
                τ₂ = max(0.0, τ₁ + Δτ₂)  # Ensure τ₂ is non-negative
                
                if τ₂ ≤ MCTS_PARAMS.max_tax
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

function ucb_score(node::MCTSNode, parent_visits::Int)
    if node.visits == 0
        return Inf
    end
    
    exploitation = node.total_value / node.visits
    exploration = MCTS_PARAMS.exploration_constant * sqrt(log(parent_visits) / node.visits)
    
    return exploitation + exploration
end

function select_node(node::MCTSNode)
    while !isempty(node.children)
        # Check for unvisited children more explicitly
        unvisited_children = filter(child -> child.visits == 0, node.children)
        if !isempty(unvisited_children)
            return first(unvisited_children)
        end
        
        # Select child with highest UCB score
        node = argmax(child -> ucb_score(child, node.visits), node.children)
    end
    return node
end

function expand_node(node::MCTSNode)
    current_tax = isnothing(node.action) ? 0.0 : node.action.τ_current
    valid_actions = get_valid_actions(current_tax)
    
    for action in valid_actions
        next_state = step_environment(node.state, action, false)  # Debug prints off
        child = MCTSNode(next_state, action, node)
        push!(node.children, child)
    end
end

function simulate(node::MCTSNode, max_depth::Int)
    if max_depth == 0
        return evaluate_state(node.state)
    end
    
    current_tax = isnothing(node.action) ? 0.0 : node.action.τ_current
    valid_actions = get_valid_actions(current_tax)
    
    if isempty(valid_actions)
        return evaluate_state(node.state)
    end
    
    action = rand(valid_actions)
    next_state = step_environment(node.state, action, false)  # Debug prints off
    
    return evaluate_state(node.state) + 
           MCTS_PARAMS.discount_factor * simulate(MCTSNode(next_state), max_depth - 1)
end

function backpropagate(node::MCTSNode, value::Float64)
    while !isnothing(node)
        node.visits += 1
        node.total_value += value
        node = node.parent
    end
end

# Main MCTS Algorithm with Parallel Processing
function mcts_search(root_state::PolicyState, n_iterations::Int, max_depth::Int)
    root = MCTSNode(root_state)
    
    # Process iterations in batches
    for batch_start in 1:BATCH_SIZE:n_iterations
        batch_end = min(batch_start + BATCH_SIZE - 1, n_iterations)
        batch_results = Vector{Tuple{MCTSNode, Float64}}(undef, batch_end - batch_start + 1)
        
        # Parallel batch processing
        @threads for i in 1:(batch_end - batch_start + 1)
            # Selection
            node = select_node(root)
            
            # Expansion
            if node.visits > 0
                expand_node(node)
                node = isempty(node.children) ? node : rand(node.children)
            end
            
            # Simulation
            value = simulate(node, max_depth)
            
            # Store results for batch update
            batch_results[i] = (node, value)
        end
        
        # Sequential batch update to maintain tree consistency
        for (node, value) in batch_results
            backpropagate(node, value)
        end
    end
    
    # Return best action based on average value
    return argmax(child -> child.total_value / child.visits, root.children).action
end

# Evaluation Functions
function evaluate_state(state::PolicyState)
    # Economic output minus climate damages
    output = state.economic_state["Y_t"]
    damages = damage_function(state.emissions, state.θ_mean)
    return output - damages
end

function damage_function(emissions::Float64, θ::Float64)
    # Quadratic damage function
    return θ * emissions^2
end

# Add step_environment function
function step_environment(state::PolicyState, action::PolicyAction, debug_print::Bool)
    # Create PolicyExpectations for RANK model
    tax_expectations = form_tax_expectations(
        action.τ_current,
        action.τ_announced,
        state.technology_params["μ_eta"],
        state.technology_params["σ_eta"],
        state.credibility
    )
    
    try
        # Get RANK equilibrium
        rank_equilibrium = compute_equilibrium(tax_expectations)
        
        # Update emissions
        new_emissions = state.emissions + rank_equilibrium["η_t"] * rank_equilibrium["Y_t"]
        
        # Update damage beliefs
        θ_new = state.θ_mean
        θ_std_new = state.θ_std * 0.95  # Simple uncertainty reduction
        
        # Return new state
        return PolicyState(
            state.time + 1,
            rank_equilibrium,
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

# Add sequential version for comparison
function mcts_search_sequential(root_state::PolicyState, n_iterations::Int, max_depth::Int)
    root = MCTSNode(root_state)
    
    for _ in 1:n_iterations
        # Selection
        node = select_node(root)
        
        # Expansion
        if node.visits > 0
            expand_node(node)
            node = isempty(node.children) ? node : rand(node.children)
        end
        
        # Simulation
        value = simulate(node, max_depth)
        
        # Backpropagation
        backpropagate(node, value)
    end
    
    return argmax(child -> child.total_value / child.visits, root.children).action
end

# Modify run_mcts_example to include benchmarking
function run_mcts_example()
    # Initialize with RANK model parameters
    initial_tech_params = Dict(
        "μ_A" => μ_A,
        "μ_eta" => μ_eta,
        "σ_A" => σ_A,
        "σ_eta" => σ_eta,
        "ρ" => ρ
    )
    
    # Run initial equilibrium to get starting economic state
    initial_expectations = form_tax_expectations(0.0, 0.0, μ_eta, σ_eta, 0.8)
    initial_econ_state = compute_equilibrium(initial_expectations)
    
    initial_state = PolicyState(
        0,
        initial_econ_state,
        0.0,
        0.001,
        0.0005,
        Float64[],
        0.8,
        initial_tech_params
    )
    
    println("\nBenchmarking Sequential MCTS...")
    sequential_time = @elapsed begin
        sequential_action = mcts_search_sequential(initial_state, 1400, 5)
    end
    
    println("\nBenchmarking Parallel MCTS...")
    parallel_time = @elapsed begin
        parallel_action = mcts_search(initial_state, 1400, 5)
    end
    
    println("\nPerformance Comparison:")
    println("Sequential Time: $(round(sequential_time, digits=2)) seconds")
    println("Parallel Time: $(round(parallel_time, digits=2)) seconds")
    println("Speedup: $(round(sequential_time/parallel_time, digits=2))x")
    
    return parallel_action
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_mcts_example()
end
