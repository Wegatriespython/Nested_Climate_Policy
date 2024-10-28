# Import necessary packages
using Random
using Distributions

# Import RANK model
include("State_function_RANK.jl")

# Copy the relevant structures from MCTS
struct PolicyState
    time::Int
    economic_state::Dict{String, Float64}
    emissions::Float64
    θ_mean::Float64    
    θ_std::Float64     
    tax_history::Vector{Float64}
    credibility::Float64
    technology_params::Dict{String, Float64}
end

struct PolicyAction
    τ_current::Float64
    τ_announced::Float64
end

function test_exact_mcts_sequence()
    # Initialize with exact same parameters as MCTS
    initial_tech_params = Dict(
        "μ_A" => μ_A,
        "μ_eta" => μ_eta,
        "σ_A" => σ_A,
        "σ_eta" => σ_eta,
        "ρ" => ρ
    )
    
    # Initialize with exact same starting state as MCTS
    initial_expectations = form_tax_expectations(0.0, 0.0, μ_eta, σ_eta, 0.8)
    initial_econ_state = compute_equilibrium(initial_expectations)
    
    current_state = PolicyState(
        0,
        initial_econ_state,
        0.0,
        0.001,  # Same θ_mean as MCTS
        0.0005, # Same θ_std as MCTS
        Float64[],
        0.8,
        initial_tech_params
    )
    
    # The exact sequence that led to error
    tax_sequence = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.0, 0.0)
    ]
    
    println("Starting exact MCTS sequence test...")
    
    for (i, (current_tax, announced_tax)) in enumerate(tax_sequence)
        println("\nStep $i:")
        println("Current state:")
        println("  Time: $(current_state.time)")
        println("  Emissions: $(current_state.emissions)")
        println("  Tax history: $(current_state.tax_history)")
        println("  θ_mean: $(current_state.θ_mean)")
        println("  θ_std: $(current_state.θ_std)")
        
        action = PolicyAction(current_tax, announced_tax)
        
        try
            # Form expectations exactly as MCTS does
            tax_expectations = form_tax_expectations(
                action.τ_current,
                action.τ_announced,
                current_state.technology_params["μ_eta"],
                current_state.technology_params["σ_eta"],
                current_state.credibility
            )
            
            # Get equilibrium
            rank_equilibrium = compute_equilibrium(tax_expectations)
            
            # Update state exactly as MCTS does
            new_emissions = current_state.emissions + rank_equilibrium["η_t"] * rank_equilibrium["Y_t"]
            θ_new = current_state.θ_mean
            θ_std_new = current_state.θ_std * 0.95
            
            # Create new state
            current_state = PolicyState(
                current_state.time + 1,
                rank_equilibrium,
                new_emissions,
                θ_new,
                θ_std_new,
                vcat(current_state.tax_history, action.τ_current),
                current_state.credibility,
                current_state.technology_params
            )
            
            # After step 3, try the simulation paths that caused error
            if i == 3
                println("\nTrying simulation paths from final state...")
                for _ in 1:10
                    future_tax = rand([0.0, 0.05, 0.1])
                    future_announced = rand([0.0, 0.05, 0.1])
                    
                    println("\nSimulating with:")
                    println("  Future tax: $future_tax")
                    println("  Future announced: $future_announced")
                    
                    # Try exact same state transition as MCTS simulate function
                    future_action = PolicyAction(future_tax, future_announced)
                    future_expectations = form_tax_expectations(
                        future_action.τ_current,
                        future_action.τ_announced,
                        current_state.technology_params["μ_eta"],
                        current_state.technology_params["σ_eta"],
                        current_state.credibility
                    )
                    compute_equilibrium(future_expectations)
                end
            end
            
        catch e
            println("\nError occurred at step $i")
            println("Full state at error:")
            println("  Time: $(current_state.time)")
            println("  Emissions: $(current_state.emissions)")
            println("  Tax history: $(current_state.tax_history)")
            println("  θ_mean: $(current_state.θ_mean)")
            println("  θ_std: $(current_state.θ_std)")
            println("  Current action: ($current_tax, $announced_tax)")
            rethrow(e)
        end
    end
end

# Run with same seed
Random.seed!(1234)
test_exact_mcts_sequence()
