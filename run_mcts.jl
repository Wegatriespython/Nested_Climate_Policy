using Random
# First include parameters
include("model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

# Then include model core
include("model_core_partial_investment.jl")

# Finally include MCTS implementation
include("mcts_better_parallelisation.jl")

# Example of creating custom parameters
const ALTERNATIVE_PARAMS = ModelParameters(
    β = 0.98,
    σ = 1.5,
    tax_revenue_weight = 0.7,
    discount_factor = 0.95
)

# Example of how to use it in your code
function run_policy_analysis()
    # Using default parameters
    result1 = run_single_policy_search(DEFAULT_PARAMS)
    
    # Using custom parameters
    custom_params = ModelParameters(
        exploration_constant = 2.5,
        tax_revenue_weight = 0.8,
        batch_size = 78,
        discount_factor = 0.97
    )
    result2 = run_single_policy_search(custom_params)
    
    # Using alternative preset
    result3 = run_single_policy_search(ALTERNATIVE_PARAMS)
end

function run_single_policy_search(params::ModelParameters = DEFAULT_PARAMS)
    # Initialize with default parameters
    initial_state = initialize_mcts(
        initial_tax = 0.0,    
        announced_tax = 0.0,    
        credibility = 0.8,     
        params = params
    )
    
    # Run MCTS search
    optimal_action = mcts_search(
        initial_state,
        1092,    # number of iterations
        5,      # search depth
        params
    )
    
    println("\nOptimal Policy Found:")
    println("Current Tax Rate: $(round(optimal_action.τ_current, digits=3))")
    println("Announced Tax Rate: $(round(optimal_action.τ_announced, digits=3))")
    
    return optimal_action
end

function run_policy_sequence(n_periods::Int, params::ModelParameters = DEFAULT_PARAMS)
    policy_sequence = Vector{PolicyAction}()
    # Fix the initialize_mcts call by providing all default values
    current_state = initialize_mcts(
        initial_tax = 0.0,
        announced_tax = 0.0,
        credibility = 0.8,
        params = params
    )
    
    println("\nComputing optimal policy sequence...")
    
    for period in 1:n_periods
        println("\nPeriod $period:")
        
        optimal_action = mcts_search(
            current_state,
            1092,    # iterations
            5,      # depth
            params
        )
        
        push!(policy_sequence, optimal_action)
        current_state = step_environment(current_state, optimal_action, true, params)
        
        println("  Current Tax: $(round(optimal_action.τ_current, digits=3))")
        println("  Announced Tax: $(round(optimal_action.τ_announced, digits=3))")
        println("  Emissions Level: $(round(current_state.emissions, digits=3))")
    end
    
    return policy_sequence
end

function run_credibility_comparison(params::ModelParameters = DEFAULT_PARAMS)
    credibility_levels = [0.2, 0.5, 0.8]
    results = Dict{Float64, PolicyAction}()
    
    println("\nComparing policies under different credibility levels...")
    
    for cred in credibility_levels
        initial_state = initialize_mcts(
            initial_tax = 0.0,
            announced_tax = 0.0,
            credibility = cred,
            params = params
        )
        
        optimal_action = mcts_search(
            initial_state,
            1092,    # iterations
            5,      # depth
            params
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
