# Import necessary packages
using Distributions
using Optim
using ForwardDiff
using NLsolve
using Plots
using Random

# Import the ModelParametersModule and use its exports explicitly
include("../model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

# Global constants
const N_SIMULATIONS = 1  # Number of simulations for Monte Carlo sampling
const DEBUG_PRINTS = false  # Control debug print statements

# Set random seed for reproducibility
Random.seed!(1234)

# 1. Structures and Types
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

# 2. Use ModelParameters instead of hardcoded values
function get_params()
    return DEFAULT_PARAMS
end

# Define the skill distribution F(θ) using ModelParametersModule
const θ_values = rand(
    Uniform(DEFAULT_PARAMS.θ_min, DEFAULT_PARAMS.θ_max), 
    DEFAULT_PARAMS.n_agents
)

# 3. Distribution Setup
# Define the joint distribution of (A_t, η_t)
function setup_distributions(params::ModelParameters = DEFAULT_PARAMS)
    cov = params.ρ * params.σ_A * params.σ_eta
    Σ = [params.σ_A^2 cov; cov params.σ_eta^2]
    return MvNormal([params.μ_A, params.μ_eta], Σ)
end

const joint_dist = setup_distributions()

# 4. Core Functions
# Update function signatures to include params parameter
function evolve_technology(time::Int, base_params::Dict{String, Float64}, 
                         is_static::Bool=true, params::ModelParameters = DEFAULT_PARAMS)
    if is_static
        return base_params
    else
        # Technology improves over time
        improvement_rate = 0.05  # 5% improvement per period
        new_params = copy(base_params)
        new_params["μ_eta"] *= (1 - improvement_rate)^time  # Mean efficiency improves
        new_params["σ_eta"] *= 0.95^time  # Dispersion decreases
        return new_params
    end
end

function adjustment_cost(Δη, θ_values, params::ModelParameters = DEFAULT_PARAMS)
    φ = params.γ * Δη^2
    ψ = mean(1.0 ./ θ_values)
    return φ * ψ
end

function optimal_technology_choice(current_η::Float64, tax_expectations::PolicyExpectations, 
                                 params::ModelParameters = DEFAULT_PARAMS)
    function technology_cost(Δη)
        future_η = current_η + Δη
        tax_cost = tax_expectations.τ_announced * future_η
        adj_cost = adjustment_cost(Δη, θ_values, params)
        return tax_cost + adj_cost
    end
    
    result = optimize(technology_cost, -0.5, 0.5)
    return result.minimizer
end

function labor_supply(w_t, C_t, params::ModelParameters = DEFAULT_PARAMS)
    return (C_t^(-params.σ) * w_t / params.χ)^(1 / params.ν)
end

# Update firm_decision to use params
function firm_decision(w_t::Float64, r_t::Float64, tax_expectations::PolicyExpectations, 
                      A_eta_samples, params::ModelParameters = DEFAULT_PARAMS)
    n = size(A_eta_samples, 1)
    profits = zeros(n)
    K_t_vals = zeros(n)
    L_t_vals = zeros(n)
    
    for i in 1:n
        A_t, η_t = A_eta_samples[i, :]
        A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
        expected_Δη = optimal_technology_choice(η_t, tax_expectations, params)
        
        try
            # Add diagnostic prints before critical calculation
            K_L_ratio = (params.α / (1 - params.α)) * (w_t / r_t)
            
            # This is where the error occurs
            L_t = ((A_eff * K_L_ratio^params.α) / w_t)^(1 / (1 - params.α))
            
            K_t = K_L_ratio * L_t
            Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
            
            adjustment_costs = adjustment_cost(expected_Δη, θ_values, params)
            profits[i] = Y_t - w_t * L_t - r_t * K_t - 
                        tax_expectations.τ_current * η_t * Y_t - 
                        adjustment_costs
            
            K_t_vals[i] = K_t
            L_t_vals[i] = L_t
            
        catch e
            if e isa DomainError
                println("\nCaught DomainError in firm_decision:")
                println("  A_t: $A_t")
                println("  η_t: $η_t")
                println("  A_eff: $A_eff")
                println("  w_t: $w_t")
                println("  r_t: $r_t")
                println("  K_L_ratio: $K_L_ratio")
                println("  τ_current: $(tax_expectations.τ_current)")
                println("  τ_announced: $(tax_expectations.τ_announced)")
                println("  Expression value: $((A_eff * K_L_ratio^params.α) / w_t)")
                println("  Exponent: $(1 / (1 - params.α))")
            end
            rethrow(e)
        end
    end
    
    max_idx = argmax(profits)
    return A_eta_samples[max_idx, 1], A_eta_samples[max_idx, 2], 
           K_t_vals[max_idx], L_t_vals[max_idx]
end

# Update compute_equilibrium to use params
function compute_equilibrium(tax_expectations::PolicyExpectations, 
                           params::ModelParameters = DEFAULT_PARAMS)
    prices_guess = [1.0, 0.05]
    
    function equilibrium_conditions(prices)
        w_t, r_t = prices
        
        # Create tech params dict for sampling
        tech_params = Dict(
            "μ_A" => params.μ_A,
            "μ_eta" => tax_expectations.η_mean,
            "σ_A" => params.σ_A,
            "σ_eta" => tax_expectations.η_std,
            "ρ" => params.ρ
        )
        
        A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
        A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples, params)
        A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
        Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
        C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params)
        L_supply = labor_supply(w_t, C_0, params)
        
        return [L_supply - L_t, K_t - K_t]
    end
    
    sol = nlsolve(equilibrium_conditions, prices_guess)
    w_t, r_t = sol.zero
    
    tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => tax_expectations.η_mean,
        "σ_A" => params.σ_A,
        "σ_eta" => tax_expectations.η_std,
        "ρ" => params.ρ
    )
    
    A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
    A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples, params)
    A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
    Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
    C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params)
    
    # Modify debug prints section
    if DEBUG_PRINTS
        println("\nEquilibrium Debug:")
        println("  Productivity (A_t): $(round(A_t, digits=4))")
        println("  Technology (η_t): $(round(η_t, digits=4))")
        println("  Effective TFP (A_eff): $(round(A_eff, digits=4))")
        println("  Capital (K_t): $(round(K_t, digits=4))")
        println("  Labor (L_t): $(round(L_t, digits=4))")
        println("  Tax effect (1 - τ*η): $(round(1 - tax_expectations.τ_current * η_t, digits=4))")
        println("  K^α * L^(1-α): $(round(K_t^params.α * L_t^(1-params.α), digits=4))")
        println("  Adjustment costs: $(round(adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params), digits=4))")
    end
    
    return Dict(
        "w_t" => w_t, "r_t" => r_t, "A_t" => A_t,
        "η_t" => η_t, "K_t" => K_t, "L_t" => L_t,
        "Y_t" => Y_t, "C_0" => C_0
    )
end

# Update state_transition to use params
function state_transition(current_state::State, policy_action::Tuple{Float64, Float64}, 
                         params::ModelParameters = DEFAULT_PARAMS)
    τ_current, τ_announced = policy_action
    credibility = 0.8
    
    # Get current technology parameters
    η_mean = current_state.technology_params["μ_eta"]
    η_std = current_state.technology_params["σ_eta"]
    
    tax_expectations = form_tax_expectations(
        τ_current, τ_announced, η_mean, η_std, credibility
    )
    
    # Use technology params in sample_A_eta
    A_eta_samples = sample_A_eta(N_SIMULATIONS, current_state.technology_params)
    rank_equilibrium = compute_equilibrium(tax_expectations, params)
    
    new_emissions = current_state.emissions + 
                   rank_equilibrium["η_t"] * rank_equilibrium["Y_t"]
    
    new_economic_state = copy(rank_equilibrium)
    new_tax_history = vcat(current_state.tax_history, τ_current)
    
    return State(
        current_state.time + 1,
        new_economic_state,
        new_emissions,
        new_tax_history,
        current_state.technology_params
    )
end

# 5. Economic Agents
function sample_A_eta(n, tech_params::Dict{String, Float64})
    μ_A = tech_params["μ_A"]
    μ_eta = tech_params["μ_eta"]
    σ_A = tech_params["σ_A"]
    σ_eta = tech_params["σ_eta"]
    ρ = tech_params["ρ"]
    
    # Convert normal parameters to lognormal parameters
    # For lognormal: E[X] = exp(μ + σ²/2), Var[X] = (exp(σ²) - 1)exp(2μ + σ²)
    μ_log_A = log(μ_A) - 0.5 * log(1 + (σ_A/μ_A)^2)
    σ_log_A = sqrt(log(1 + (σ_A/μ_A)^2))
    
    μ_log_eta = log(μ_eta) - 0.5 * log(1 + (σ_eta/μ_eta)^2)
    σ_log_eta = sqrt(log(1 + (σ_eta/μ_eta)^2))
    
    # Create bivariate lognormal using correlation in log space
    Σ_log = [σ_log_A^2 ρ*σ_log_A*σ_log_eta; 
             ρ*σ_log_A*σ_log_eta σ_log_eta^2]
    
    dist = MvLogNormal([μ_log_A, μ_log_eta], Σ_log)
    
    return rand(dist, n)'
end

function form_tax_expectations(τ_current::Float64, τ_announced::Float64, 
                             η_mean::Float64, η_std::Float64, credibility::Float64)
    return PolicyExpectations(τ_current, τ_announced, η_mean, η_std, credibility)
end

# 6. Equilibrium
function compute_equilibrium(tax_expectations::PolicyExpectations, 
                           params::ModelParameters = DEFAULT_PARAMS)
    prices_guess = [1.0, 0.05]
    
    function equilibrium_conditions(prices)
        w_t, r_t = prices
        
        # Create tech params dict for sampling
        tech_params = Dict(
            "μ_A" => params.μ_A,
            "μ_eta" => tax_expectations.η_mean,
            "σ_A" => params.σ_A,
            "σ_eta" => tax_expectations.η_std,
            "ρ" => params.ρ
        )
        
        A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
        A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples, params)
        A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
        Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
        C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params)
        L_supply = labor_supply(w_t, C_0, params)
        
        return [L_supply - L_t, K_t - K_t]
    end
    
    sol = nlsolve(equilibrium_conditions, prices_guess)
    w_t, r_t = sol.zero
    
    tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => tax_expectations.η_mean,
        "σ_A" => params.σ_A,
        "σ_eta" => tax_expectations.η_std,
        "ρ" => params.ρ
    )
    
    A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
    A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples, params)
    A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
    Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
    C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params)
    
    # Modify debug prints section
    if DEBUG_PRINTS
        println("\nEquilibrium Debug:")
        println("  Productivity (A_t): $(round(A_t, digits=4))")
        println("  Technology (η_t): $(round(η_t, digits=4))")
        println("  Effective TFP (A_eff): $(round(A_eff, digits=4))")
        println("  Capital (K_t): $(round(K_t, digits=4))")
        println("  Labor (L_t): $(round(L_t, digits=4))")
        println("  Tax effect (1 - τ*η): $(round(1 - tax_expectations.τ_current * η_t, digits=4))")
        println("  K^α * L^(1-α): $(round(K_t^params.α * L_t^(1-params.α), digits=4))")
        println("  Adjustment costs: $(round(adjustment_cost(η_t - tax_expectations.η_mean, θ_values, params), digits=4))")
    end
    
    return Dict(
        "w_t" => w_t, "r_t" => r_t, "A_t" => A_t,
        "η_t" => η_t, "K_t" => K_t, "L_t" => L_t,
        "Y_t" => Y_t, "C_0" => C_0
    )
end

# 7. State Transition
function state_transition(current_state::State, policy_action::Tuple{Float64, Float64}, 
                         params::ModelParameters = DEFAULT_PARAMS)
    τ_current, τ_announced = policy_action
    credibility = 0.8
    
    # Get current technology parameters
    η_mean = current_state.technology_params["μ_eta"]
    η_std = current_state.technology_params["σ_eta"]
    
    tax_expectations = form_tax_expectations(
        τ_current, τ_announced, η_mean, η_std, credibility
    )
    
    # Use technology params in sample_A_eta
    A_eta_samples = sample_A_eta(N_SIMULATIONS, current_state.technology_params)
    rank_equilibrium = compute_equilibrium(tax_expectations, params)
    
    new_emissions = current_state.emissions + 
                   rank_equilibrium["η_t"] * rank_equilibrium["Y_t"]
    
    new_economic_state = copy(rank_equilibrium)
    new_tax_history = vcat(current_state.tax_history, τ_current)
    
    return State(
        current_state.time + 1,
        new_economic_state,
        new_emissions,
        new_tax_history,
        current_state.technology_params
    )
end

# 8. Test Function
function run_test(params::ModelParameters = DEFAULT_PARAMS)
    # Initialize with technology parameters
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
    
    # Fixed plotting code
    times = 1:length(states)-1  # Changed from 0:length(states)-1
    outputs = Float64[]
    emissions = Float64[]
    
    # Skip the initial state which has empty economic_state
    for state in states[2:end]
        push!(outputs, state.economic_state["Y_t"])
        push!(emissions, state.emissions)
    end
    
    p1 = plot(times, outputs, 
         label="Output", xlabel="Time", ylabel="Output",
         title="Economic Output Over Time",
         marker=:circle)  # Added marker for better visibility
    
    p2 = plot(times, emissions,
         label="Cumulative Emissions", xlabel="Time", ylabel="Emissions",
         title="Emissions Over Time",
         marker=:circle)  # Added marker for better visibility
    
    final_plot = plot(p1, p2, layout=(2,1), size=(800,600))
    savefig(final_plot, "simulation_results.png")
    
    return states
end

# Run test if file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    states = run_test()
end

function run_comparison_test(params::ModelParameters = DEFAULT_PARAMS)
    # Initial technology parameters
    base_tech_params = Dict(
        "μ_A" => 1.0,    # Mean productivity
        "μ_eta" => 1.0,  # Mean carbon intensity
        "σ_A" => 0.2,    # Std dev of productivity
        "σ_eta" => 0.2,  # Std dev of carbon intensity
        "ρ" => 0.5       # Correlation coefficient
    )
    
    # Initialize states for both scenarios
    static_state = State(
        0,
        Dict{String, Float64}(),
        0.0,
        Float64[],
        copy(base_tech_params)
    )
    
    evolving_state = State(
        0,
        Dict{String, Float64}(),
        0.0,
        Float64[],
        copy(base_tech_params)
    )
    
    # Policy actions
    policy_actions = [
        (0.05, 0.06),
        (0.06, 0.07),
        (0.07, 0.07)
    ]
    
    # Run simulations
    static_states = [static_state]
    evolving_states = [evolving_state]
    
    println("Starting comparison simulation...")
    
    for (i, action) in enumerate(policy_actions)
        println("\nPeriod $i:")
        
        # Static technology case
        static_state.technology_params = evolve_technology(i, base_tech_params, true, params)
        new_static = state_transition(static_state, action, params)
        push!(static_states, new_static)
        
        # Evolving technology case
        evolving_state.technology_params = evolve_technology(i, base_tech_params, false, params)
        new_evolving = state_transition(evolving_state, action, params)
        push!(evolving_states, new_evolving)
        
        # Print comparison
        println("\nStatic Technology:")
        println("Output: $(new_static.economic_state["Y_t"])")
        println("Emissions: $(new_static.emissions)")
        println("Carbon Intensity: $(new_static.economic_state["η_t"])")
        
        println("\nEvolving Technology:")
        println("Output: $(new_evolving.economic_state["Y_t"])")
        println("Emissions: $(new_evolving.emissions)")
        println("Carbon Intensity: $(new_evolving.economic_state["η_t"])")
    end
    
    # Plot comparisons
    times = 1:length(static_states)-1
    
    # Collect data
    static_outputs = [state.economic_state["Y_t"] for state in static_states[2:end]]
    static_emissions = [state.emissions for state in static_states[2:end]]
    static_intensity = [state.economic_state["η_t"] for state in static_states[2:end]]
    
    evolving_outputs = [state.economic_state["Y_t"] for state in evolving_states[2:end]]
    evolving_emissions = [state.emissions for state in evolving_states[2:end]]
    evolving_intensity = [state.economic_state["η_t"] for state in evolving_states[2:end]]
    
    # Create plots
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
    
    return static_states, evolving_states
end

# Run test if file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running comparison test...")
    static_states, evolving_states = run_comparison_test()
end

# Export necessary functions
export PolicyExpectations, State, compute_equilibrium, form_tax_expectations

# Run test if file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    states = run_test()
    static_states, evolving_states = run_comparison_test()
end
