# Import necessary packages
using Distributions
using Optim
using ForwardDiff
using NLsolve
using Plots
using Random

# Import the ModelParametersModule and use its exports explicitly
include("model_parameters.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

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

# 2. Hardcoded Parameters (from Old_State_Function.jl)
# Economic parameters
const β = 0.96          # Discount factor
const σ = 2.0           # Relative risk aversion coefficient
const χ = 1.0           # Labor disutility weight
const ν = 1.0           # Inverse of Frisch elasticity
const α = 0.33          # Capital share parameter
const δ = 0.1           # Depreciation rate
const γ = 0.1           # Adjustment cost coefficient

# Joint distribution parameters
const μ_A = 1.0         # Mean productivity
const μ_eta = 1.0       # Mean carbon intensity
const σ_A = 0.2         # Std dev of productivity
const σ_eta = 0.2       # Std dev of carbon intensity
const ρ = 0.5           # Correlation coefficient

# Skill distribution parameters
const θ_min = 0.1       # Minimum skill level
const θ_max = 1.0       # Maximum skill level
const n_agents = 1000   # Number of agents/workers

const DEBUG_PRINT = false  # Add this flag
const N_SIMULATIONS = 1   # Reduce from 1000 to 200

# Define the skill distribution F(θ) using ModelParametersModule
const θ_values = rand(
    Uniform(ModelParametersModule.DEFAULT_PARAMS.θ_min, 
           ModelParametersModule.DEFAULT_PARAMS.θ_max), 
    ModelParametersModule.DEFAULT_PARAMS.n_agents
)

# Rest of the functions updated to use hardcoded parameters instead of ModelParameters

function adjustment_cost(Δη, θ_values)
    φ = γ * Δη^2
    ψ = mean(1.0 ./ θ_values)
    return φ * ψ
end

function optimal_technology_choice(current_η::Float64, tax_expectations::PolicyExpectations)
    function technology_cost(Δη)
        future_η = current_η + Δη
        tax_cost = tax_expectations.τ_announced * future_η
        adj_cost = adjustment_cost(Δη, θ_values)
        return tax_cost + adj_cost
    end
    
    result = optimize(technology_cost, -0.5, 0.5)
    return result.minimizer
end

function labor_supply(w_t, C_t)
    return (C_t^(-σ) * w_t / χ)^(1 / ν)
end

# Keep the improved error handling but use hardcoded parameters
function firm_decision(w_t::Float64, r_t::Float64, tax_expectations::PolicyExpectations, A_eta_samples)
    n = size(A_eta_samples, 1)
    profits = zeros(n)
    K_t_vals = zeros(n)
    L_t_vals = zeros(n)
    
    for i in 1:n
        A_t, η_t = A_eta_samples[i, :]
        A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
        expected_Δη = optimal_technology_choice(η_t, tax_expectations)
        
        try
            K_L_ratio = (α / (1 - α)) * (w_t / r_t)
            term1 = A_eff * K_L_ratio^α
            term2 = term1 / w_t
            exponent = 1 / (1 - α)
            
            if term2 < 0
                println("\nNegative base detected in firm_decision:")
                println("  A_t: $A_t")
                println("  η_t: $η_t")
                println("  A_eff: $A_eff")
                println("  K_L_ratio: $K_L_ratio")
                println("  term1: $term1")
                println("  term2: $term2")
                println("  w_t: $w_t")
                println("  r_t: $r_t")
                println("  τ_current: $(tax_expectations.τ_current)")
                println("  τ_announced: $(tax_expectations.τ_announced)")
                println("  exponent: $exponent")
            end
            
            L_t = ((A_eff * K_L_ratio^α) / w_t)^(1 / (1 - α))
            K_t = K_L_ratio * L_t
            Y_t = A_eff * K_t^α * L_t^(1 - α)
            
            adjustment_costs = adjustment_cost(expected_Δη, θ_values)
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
                println("  K_L_ratio: $K_L_ratio")
                println("  w_t: $w_t")
                println("  r_t: $r_t")
                println("  τ_current: $(tax_expectations.τ_current)")
                println("  τ_announced: $(tax_expectations.τ_announced)")
                println("  Expression value: $((A_eff * K_L_ratio^α) / w_t)")
                println("  Exponent: $(1 / (1 - α))")
            end
            rethrow(e)
        end
    end
    
    max_idx = argmax(profits)
    return A_eta_samples[max_idx, 1], A_eta_samples[max_idx, 2], 
           K_t_vals[max_idx], L_t_vals[max_idx]
end

function compute_equilibrium(tax_expectations::PolicyExpectations)
    prices_guess = [1.0, 0.05]
    
    function equilibrium_conditions(prices)
        try
            w_t, r_t = prices
            
            tech_params = Dict(
                "μ_A" => μ_A,
                "μ_eta" => tax_expectations.η_mean,
                "σ_A" => σ_A,
                "σ_eta" => tax_expectations.η_std,
                "ρ" => ρ
            )
            
            A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
            A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples)
            A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
            Y_t = A_eff * K_t^α * L_t^(1 - α)
            C_0 = Y_t - δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values)
            L_supply = labor_supply(w_t, C_0)
            
            residuals = [L_supply - L_t, K_t - K_t]
            
            if any(isnan, residuals) || any(isinf, residuals)
                println("\nInvalid residuals in equilibrium_conditions:")
                println("  w_t: $w_t")
                println("  r_t: $r_t")
                println("  L_supply: $L_supply")
                println("  L_t: $L_t")
                println("  K_t: $K_t")
                println("  Y_t: $Y_t")
                println("  C_0: $C_0")
            end
            
            return residuals
            
        catch e
            println("\nError in equilibrium_conditions:")
            println("  Prices: $prices")
            println("  Tax expectations: $tax_expectations")
            rethrow(e)
        end
    end
    
    sol = nlsolve(equilibrium_conditions, prices_guess)
    
    if !converged(sol)
        println("\nEquilibrium solver failed to converge:")
        println("  Final prices: $(sol.zero)")
        println("  Final residuals: $(sol.residual)")
        println("  Iterations: $(sol.iterations)")
        throw(ErrorException("Equilibrium solver did not converge"))
    end
    
    w_t, r_t = sol.zero
    tech_params = Dict(
        "μ_A" => μ_A,
        "μ_eta" => tax_expectations.η_mean,
        "σ_A" => σ_A,
        "σ_eta" => tax_expectations.η_std,
        "ρ" => ρ
    )
    
    A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
    A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples)
    A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
    Y_t = A_eff * K_t^α * L_t^(1 - α)
    C_0 = Y_t - δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values)
    
    return Dict(
        "w_t" => w_t, "r_t" => r_t, "A_t" => A_t,
        "η_t" => η_t, "K_t" => K_t, "L_t" => L_t,
        "Y_t" => Y_t, "C_0" => C_0
    )
end

# 3. Distribution Setup
# Define the joint distribution of (A_t, η_t)
cov = ρ * σ_A * σ_eta
Σ = [σ_A^2 cov; cov σ_eta^2]
joint_dist = MvNormal([μ_A, μ_eta], Σ)

# 4. Core Functions
# Add evolve_technology here, before it's used
function evolve_technology(time::Int, base_params::Dict{String, Float64}, 
                         is_static::Bool=true)
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

function sample_A_eta(n::Int, tech_params::Dict{String, Float64})
    μ_A = tech_params["μ_A"]
    μ_eta = tech_params["μ_eta"]
    σ_A = tech_params["σ_A"]
    σ_eta = tech_params["σ_eta"]
    ρ = tech_params["ρ"]
    
    μ_log_A = log(μ_A) - 0.5 * log(1 + (σ_A/μ_A)^2)
    σ_log_A = sqrt(log(1 + (σ_A/μ_A)^2))
    
    μ_log_eta = log(μ_eta) - 0.5 * log(1 + (σ_eta/μ_eta)^2)
    σ_log_eta = sqrt(log(1 + (σ_eta/μ_eta)^2))
    
    Σ_log = [σ_log_A^2 ρ*σ_log_A*σ_log_eta; 
             ρ*σ_log_A*σ_log_eta σ_log_eta^2]
    
    dist = MvLogNormal([μ_log_A, μ_log_eta], Σ_log)
    
    return rand(dist, n)'
end

function form_tax_expectations(τ_current::Float64, τ_announced::Float64, 
                             η_mean::Float64, η_std::Float64, credibility::Float64)
    return PolicyExpectations(τ_current, τ_announced, η_mean, η_std, credibility)
end

# 5. Economic Agents
function log_error(context::String, e::Exception, args...)
    println("\nError in $context:")
    for (name, value) in args
        println("  $name: $value")
    end
    println("Error: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

function safe_compute(f::Function, context::String, args...)
    try
        return f()
    catch e
        log_error(context, e, args...)
        rethrow(e)
    end
end

# 6. Equilibrium
# Update compute_equilibrium with better error handling
function compute_equilibrium(tax_expectations::PolicyExpectations, params::ModelParameters)
    prices_guess = [1.0, 0.05]
    
    function equilibrium_conditions(prices)
        try
            w_t, r_t = prices
            
            tech_params = Dict(
                "μ_A" => params.μ_A,
                "μ_eta" => tax_expectations.η_mean,
                "σ_A" => params.σ_A,
                "σ_eta" => tax_expectations.η_std,
                "ρ" => params.ρ
            )
            
            A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
            A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples)
            A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
            Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
            C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values)
            L_supply = labor_supply(w_t, C_0)
            
            residuals = [L_supply - L_t, K_t - K_t]
            
            if any(isnan, residuals) || any(isinf, residuals)
                println("\nInvalid residuals in equilibrium_conditions:")
                println("  w_t: $w_t")
                println("  r_t: $r_t")
                println("  L_supply: $L_supply")
                println("  L_t: $L_t")
                println("  K_t: $K_t")
                println("  Y_t: $Y_t")
                println("  C_0: $C_0")
            end
            
            return residuals
            
        catch e
            println("\nError in equilibrium_conditions:")
            println("  Prices: $prices")
            println("  Tax expectations: $tax_expectations")
            rethrow(e)
        end
    end
    
    sol = nlsolve(equilibrium_conditions, prices_guess)
    
    if !converged(sol)
        println("\nEquilibrium solver failed to converge:")
        println("  Final prices: $(sol.zero)")
        println("  Final residuals: $(sol.residual)")
        println("  Iterations: $(sol.iterations)")
        throw(ErrorException("Equilibrium solver did not converge"))
    end
    
    # Compute final equilibrium values
    w_t, r_t = sol.zero
    tech_params = Dict(
        "μ_A" => params.μ_A,
        "μ_eta" => tax_expectations.η_mean,
        "σ_A" => params.σ_A,
        "σ_eta" => tax_expectations.η_std,
        "ρ" => params.ρ
    )
    
    A_eta_samples = sample_A_eta(N_SIMULATIONS, tech_params)
    A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, tax_expectations, A_eta_samples)
    A_eff = (1 - tax_expectations.τ_current * η_t) * A_t
    Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
    C_0 = Y_t - params.δ * K_t - adjustment_cost(η_t - tax_expectations.η_mean, θ_values)
    
    return Dict(
        "w_t" => w_t, "r_t" => r_t, "A_t" => A_t,
        "η_t" => η_t, "K_t" => K_t, "L_t" => L_t,
        "Y_t" => Y_t, "C_0" => C_0
    )
end

# 7. State Transition
function state_transition(current_state::State, policy_action::Tuple{Float64, Float64})
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
    rank_equilibrium = compute_equilibrium(tax_expectations, ModelParametersModule.DEFAULT_PARAMS)
    
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
function run_test()
    # Initialize with technology parameters
    initial_tech_params = Dict(
        "μ_A" => ModelParametersModule.DEFAULT_PARAMS.μ_A,
        "μ_eta" => ModelParametersModule.DEFAULT_PARAMS.μ_eta,
        "σ_A" => ModelParametersModule.DEFAULT_PARAMS.σ_A,
        "σ_eta" => ModelParametersModule.DEFAULT_PARAMS.σ_eta,
        "ρ" => ModelParametersModule.DEFAULT_PARAMS.ρ
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
        
        current_state = state_transition(current_state, action)
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

function run_comparison_test()
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
        static_state.technology_params = evolve_technology(i, base_tech_params, true)
        new_static = state_transition(static_state, action)
        push!(static_states, new_static)
        
        # Evolving technology case
        evolving_state.technology_params = evolve_technology(i, base_tech_params, false)
        new_evolving = state_transition(evolving_state, action)
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
export compute_equilibrium, form_tax_expectations

