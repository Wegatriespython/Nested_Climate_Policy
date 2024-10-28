using Random
include("mcts_parallel.jl")

function run_single_policy_search()
    # Initialize with default parameters
    initial_state = initialize_mcts(
        0.0,    # initial tax
        0.0,    # announced tax
        0.8     # credibility
    )
    
    # Run MCTS search
    optimal_action = mcts_search(
        initial_state,
        1000,   # number of iterations
        5       # search depth
    )
    
    println("\nOptimal Policy Found:")
    println("Current Tax Rate: $(round(optimal_action.τ_current, digits=3))")
    println("Announced Tax Rate: $(round(optimal_action.τ_announced, digits=3))")
    
    return optimal_action
end

function run_policy_sequence(n_periods::Int)
    # Store sequence of policies
    policy_sequence = Vector{PolicyAction}()
    current_state = initialize_mcts()
    
    println("\nComputing optimal policy sequence...")
    
    for period in 1:n_periods
        println("\nPeriod $period:")
        
        optimal_action = mcts_search(
            current_state,
            1000,   # iterations
            5       # depth
        )
        
        push!(policy_sequence, optimal_action)
        
        # Step environment forward
        current_state = step_environment(current_state, optimal_action, false)
        
        println("  Current Tax: $(round(optimal_action.τ_current, digits=3))")
        println("  Announced Tax: $(round(optimal_action.τ_announced, digits=3))")
        println("  Emissions Level: $(round(current_state.emissions, digits=3))")
    end
    
    return policy_sequence
end

function run_credibility_comparison()
    credibility_levels = [0.2, 0.5, 0.8]
    results = Dict{Float64, PolicyAction}()
    
    println("\nComparing policies under different credibility levels...")
    
    for cred in credibility_levels
        initial_state = initialize_mcts(0.0, 0.0, cred)
        
        optimal_action = mcts_search(
            initial_state,
            1000,   # iterations
            5       # depth
        )
        
        results[cred] = optimal_action
        
        println("\nCredibility Level: $cred")
        println("  Current Tax: $(round(optimal_action.τ_current, digits=3))")
        println("  Announced Tax: $(round(optimal_action.τ_announced, digits=3))")
    end
    
    return results
end

# Set random seed for reproducibility
Random.seed!(123)

# Choose which analysis to run
println("Running single policy search...")
single_result = run_single_policy_search()

println("\nRunning policy sequence...")
sequence_result = run_policy_sequence(3)

println("\nRunning credibility comparison...")
credibility_results = run_credibility_comparison()
