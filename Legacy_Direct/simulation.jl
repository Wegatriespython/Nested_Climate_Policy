using Plots
include("equilibrium_solver.jl")

function evolve_technology(time::Int, base_params::Dict{String, Float64}, 
                         is_static::Bool=true, params::ModelParameters = DEFAULT_PARAMS)
    if is_static
        return base_params
    else
        improvement_rate = 0.05  # 5% improvement per period
        new_params = copy(base_params)
        new_params["μ_eta"] *= (1 - improvement_rate)^time  # Mean efficiency improves
        new_params["σ_eta"] *= 0.95^time  # Dispersion decreases
        return new_params
    end
end

function state_transition(current_state::State, policy_action::Tuple{Float64, Float64}, 
                         params::ModelParameters = DEFAULT_PARAMS)
    τ_current, τ_announced = policy_action
    credibility = 0.8
    
    η_mean = current_state.technology_params["μ_eta"]
    η_std = current_state.technology_params["σ_eta"]
    
    tax_expectations = form_tax_expectations(
        τ_current, τ_announced, η_mean, η_std, credibility
    )
    
    A_eta_samples = sample_A_eta(N_SIMULATIONS, current_state.technology_params)
    rank_equilibrium = compute_equilibrium(tax_expectations, params)
    
    new_emissions = current_state.emissions + 
                   rank_equilibrium["η_t"] * rank_equilibrium["Y_t"]
    
    return State(
        current_state.time + 1,
        copy(rank_equilibrium),
        new_emissions,
        vcat(current_state.tax_history, τ_current),
        current_state.technology_params
    )
end

function run_test(params::ModelParameters = DEFAULT_PARAMS)
    initial_tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => params.μ_eta,
        "σ_A" => params.σ_A,
        "σ_eta" => params.σ_eta,
        "ρ" => params.ρ
    )
    
    initial_state = State(
        0,
        Dict{String, Float64}(),
        0.0,
        Float64[],
        initial_tech_params
    )
    
    policy_actions = [
        (0.05, 0.06),
        (0.06, 0.07),
        (0.07, 0.07)
    ]
    
    current_state = initial_state
    states = [current_state]
    
    println("Starting simulation...")
    for (i, action) in enumerate(policy_actions)
        println("\nPeriod $i:")
        println("Policy Action: Current τ = $(action[1]), Announced τ = $(action[2])")
        
        current_state = state_transition(current_state, action, params)
        push!(states, current_state)
        
        println("Output: $(current_state.economic_state["Y_t"])")
        println("Emissions: $(current_state.emissions)")
        println("Carbon Intensity: $(current_state.economic_state["η_t"])")
    end
    
    plot_results(states)
    return states
end

function plot_results(states)
    times = 1:length(states)-1
    outputs = [state.economic_state["Y_t"] for state in states[2:end]]
    emissions = [state.emissions for state in states[2:end]]
    
    p1 = plot(times, outputs, 
         label="Output", xlabel="Time", ylabel="Output",
         title="Economic Output Over Time",
         marker=:circle)
    
    p2 = plot(times, emissions,
         label="Cumulative Emissions", xlabel="Time", ylabel="Emissions",
         title="Emissions Over Time",
         marker=:circle)
    
    final_plot = plot(p1, p2, layout=(2,1), size=(800,600))
    savefig(final_plot, "simulation_results.png")
end

function run_comparison_test(params::ModelParameters = DEFAULT_PARAMS)
    base_tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => params.μ_eta,
        "σ_A" => params.σ_A,
        "σ_eta" => params.σ_eta,
        "ρ" => params.ρ
    )
    
    static_state = State(0, Dict{String, Float64}(), 0.0, Float64[], copy(base_tech_params))
    evolving_state = State(0, Dict{String, Float64}(), 0.0, Float64[], copy(base_tech_params))
    
    policy_actions = [(0.05, 0.06), (0.06, 0.07), (0.07, 0.07)]
    
    static_states = [static_state]
    evolving_states = [evolving_state]
    
    println("Starting comparison simulation...")
    
    for (i, action) in enumerate(policy_actions)
        println("\nPeriod $i:")
        
        static_state.technology_params = evolve_technology(i, base_tech_params, true, params)
        evolving_state.technology_params = evolve_technology(i, base_tech_params, false, params)
        
        new_static = state_transition(static_state, action, params)
        new_evolving = state_transition(evolving_state, action, params)
        
        push!(static_states, new_static)
        push!(evolving_states, new_evolving)
        
        print_comparison(new_static, new_evolving)
    end
    
    plot_comparison(static_states, evolving_states)
    return static_states, evolving_states
end

function print_comparison(static_state::State, evolving_state::State)
    println("\nStatic Technology:")
    println("Output: $(static_state.economic_state["Y_t"])")
    println("Emissions: $(static_state.emissions)")
    println("Carbon Intensity: $(static_state.economic_state["η_t"])")
    
    println("\nEvolving Technology:")
    println("Output: $(evolving_state.economic_state["Y_t"])")
    println("Emissions: $(evolving_state.emissions)")
    println("Carbon Intensity: $(evolving_state.economic_state["η_t"])")
end

function plot_comparison(static_states, evolving_states)
    times = 1:length(static_states)-1
    
    static_outputs = [state.economic_state["Y_t"] for state in static_states[2:end]]
    static_emissions = [state.emissions for state in static_states[2:end]]
    static_intensity = [state.economic_state["η_t"] for state in static_states[2:end]]
    
    evolving_outputs = [state.economic_state["Y_t"] for state in evolving_states[2:end]]
    evolving_emissions = [state.emissions for state in evolving_states[2:end]]
    evolving_intensity = [state.economic_state["η_t"] for state in evolving_states[2:end]]
    
    p1 = plot(times, [static_outputs evolving_outputs], 
         label=["Static" "Evolving"], 
         title="Output Comparison",
         xlabel="Time", ylabel="Output")
    
    p2 = plot(times, [static_emissions evolving_emissions],
         label=["Static" "Evolving"],
         title="Emissions Comparison",
         xlabel="Time", ylabel="Cumulative Emissions")
    
    p3 = plot(times, [static_intensity evolving_intensity],
         label=["Static" "Evolving"],
         title="Carbon Intensity Comparison",
         xlabel="Time", ylabel="Carbon Intensity")
    
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800,900))
    savefig(final_plot, "technology_comparison.png")
end

# Run tests if file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    states = run_test()
    static_states, evolving_states = run_comparison_test()
end

export evolve_technology, state_transition, run_test, run_comparison_test