using JuMP
using Ipopt
using Distributions
using Random
using QuadGK
using Statistics

"""
Represents the joint distribution of productivity and carbon intensity
"""
struct TechnologyDistribution
    μ_A::Float64  # Mean productivity
    μ_η::Float64  # Mean carbon intensity
    σ_A::Float64  # Productivity std
    σ_η::Float64  # Carbon intensity std
    ρ::Float64    # Correlation coefficient
    
    function TechnologyDistribution(μ_A, μ_η, σ_A, σ_η, ρ)
        new(μ_A, μ_η, σ_A, σ_η, clamp(ρ, -1.0, 1.0))
    end
end

"""
Computes the adjustment cost given skill level θ and change in carbon intensity
"""
function adjustment_cost_density(Δη::Float64, θ::Float64, γ::Float64)
    return γ * (Δη^2) * (1/θ)
end

"""
Computes the expected adjustment cost integrated over the skill distribution
"""
function expected_adjustment_cost(Δη::Float64, γ::Float64)
    # Assume uniform skill distribution over [θ_min, θ_max] for simplicity
    θ_min, θ_max = 0.5, 2.0
    
    # Integrate adjustment cost over skill distribution
    cost, _ = quadgk(θ -> adjustment_cost_density(Δη, θ, γ), θ_min, θ_max)
    return cost / (θ_max - θ_min)  # Normalize by distribution width
end

"""
Samples technology parameters for both periods with persistence
"""
function sample_technology_parameters(
    params::ModelParameters,
    n_samples::Int = 20
)
    # Create the joint distribution
    dist = MvNormal([params.μ_A, params.μ_eta], 
                    [params.σ_A^2 params.ρ*params.σ_A*params.σ_eta;
                     params.ρ*params.σ_A*params.σ_eta params.σ_eta^2])
    
    # Storage for samples
    A_0_samples = zeros(n_samples)
    η_0_samples = zeros(n_samples)
    A_1_samples = zeros(n_samples)
    η_1_samples = zeros(n_samples)
    
    # Persistence parameters
    ρ_A = 0.8  # Productivity persistence
    ρ_η = 0.9  # Carbon intensity persistence
    
    # Generate samples
    for i in 1:n_samples
        # Period 0
        A_0, η_0 = rand(dist)
        
        # Period 1 with persistence
        A_1 = ρ_A * A_0 + (1 - ρ_A) * params.μ_A + params.σ_A * randn() * sqrt(1 - ρ_A^2)
        η_1 = ρ_η * η_0 + (1 - ρ_η) * params.μ_eta + params.σ_eta * randn() * sqrt(1 - ρ_η^2)
        
        A_0_samples[i] = A_0
        η_0_samples[i] = η_0
        A_1_samples[i] = A_1
        η_1_samples[i] = η_1
    end
    
    # Return median values as representative
    return (
        A_0 = median(A_0_samples),
        η_0 = median(η_0_samples),
        A_1 = median(A_1_samples),
        η_1 = median(η_1_samples)
    )
end

"""
Computes the general equilibrium for the two-period model
"""
function compute_equilibrium(
    policy::PolicyExpectations,
    params::ModelParameters = DEFAULT_PARAMS
)
    # Sample technology parameters
    tech_params = sample_technology_parameters(params)
    A_0, η_0 = tech_params.A_0, tech_params.η_0
    A_1, η_1 = tech_params.A_1, tech_params.η_1
    
    # Pre-compute effective TFPs with tax effects
    A_eff_0 = (1 - policy.τ_current * η_0) * A_0
    A_eff_1 = (1 - policy.τ_announced * η_1) * A_1
    
    # Print diagnostic information
    println("Technology Parameters:")
    println("  A_0: $A_0, η_0: $η_0")
    println("  A_1: $A_1, η_1: $η_1")
    println("Effective TFP:")
    println("  A_eff_0: $A_eff_0")
    println("  A_eff_1: $A_eff_1")
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    # Configure Ipopt
    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "print_level", 5)
    

    # Declare all variables with minimal bounds to prevent numerical issues
    @variables(model, begin
        w_0 >= 1e-6    # Wage
        L_0 >= 1e-6    # Labor
        C_0 >= 1e-6    # Consumption
        S_0 >= 1e-6    # Savings
        Y_0 >= 1e-6    # Output
        K_0 == params.K_init    # Capital
        adj_cost >= 0  # Adjustment cost
        
        w_1 >= 1e-6    # Wage
        -0.99 <= r_1 <= 10.0    # Interest rate (prevent r_1 = -1)
        L_1 >= 1e-6    # Labor
        C_1 >= 1e-6    # Consumption
        Y_1 >= 1e-6    # Output
        K_1 >= 1e-6    # Capital
    end)
    
    # Fix initial capital
    fix(K_0, params.K_init)
    
    # Compute adjustment cost parameter
    Δη = η_1 - η_0
    
    # Production constraints with pre-computed effective TFP
    @NLconstraint(model, Y_0 == A_eff_0 * K_0^params.α * L_0^(1-params.α))
    @NLconstraint(model, Y_1 == A_eff_1 * K_1^params.α * L_1^(1-params.α))
    
    # Adjustment cost constraint
    @NLconstraint(model, adj_cost == params.γ * Δη^2)
    
    # Budget constraints
    @NLconstraint(model, C_0 + S_0 == w_0 * L_0 - adj_cost)
    @NLconstraint(model, C_1 == w_1 * L_1 + (1 + r_1) * S_0)
    
    # Labor supply
    @NLconstraint(model, params.χ * L_0^params.ν == C_0^(-params.σ) * w_0)
    @NLconstraint(model, params.χ * L_1^params.ν == C_1^(-params.σ) * w_1)
    
    # Euler equation
    @NLconstraint(model, C_0^(-params.σ) == params.β * (1 + r_1) * C_1^(-params.σ))
    
    # Capital accumulation
    @NLconstraint(model, K_1 == S_0)
    
    # Firm's FOCs
    @NLconstraint(model, w_0 == (1-params.α) * Y_0 / L_0)
    @NLconstraint(model, w_1 == (1-params.α) * Y_1 / L_1)
    @NLconstraint(model, r_1 + params.δ == params.α * Y_1 / K_1)
    
    # Relaxed stability constraints
    @constraint(model, Y_1 >= 0.5 * Y_0)  # Allow for larger output decline
    @constraint(model, L_1 >= 0.5 * L_0)  # Allow for larger labor decline
    @constraint(model, C_1 >= 0.5 * C_0)  # Allow for larger consumption decline
    
    # Objective: Maximize household utility
    @NLobjective(model, Max, 
        (C_0^(1-params.σ)/(1-params.σ) - params.χ * L_0^(1+params.ν)/(1+params.ν)) +
        params.β * (C_1^(1-params.σ)/(1-params.σ) - params.χ * L_1^(1+params.ν)/(1+params.ν)) -
        adj_cost
    )
    
    optimize!(model)
    
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        println("\nOptimization failed:")
        println("Status: $status")
        println("Solver specific status: ", raw_status(model))
        println("Objective value: ", objective_value(model))
        error("Failed to find equilibrium: $status")
    end
    
    return Dict(
        # Period 0
        "w_0" => value(w_0), "L_0" => value(L_0),
        "C_0" => value(C_0), "Y_0" => value(Y_0),
        "K_0" => value(K_0), "S_0" => value(S_0),
        "η_0" => η_0, "A_0" => A_0,
        "adj_cost" => value(adj_cost),
        
        # Period 1
        "w_1" => value(w_1), "r_1" => value(r_1),
        "L_1" => value(L_1), "C_1" => value(C_1),
        "Y_1" => value(Y_1), "K_1" => value(K_1),
        "η_1" => η_1, "A_1" => A_1,
        
        # Technology changes
        "d_eta" => Δη,        # Changed from "Δη"
        "d_A" => A_1 - A_0    # Changed from "ΔA"
    )
end

# Alternative version that accepts explicit TechnologyDistribution
function compute_equilibrium(
    tech_dist::TechnologyDistribution,
    policy::PolicyExpectations,
    params::ModelParameters = DEFAULT_PARAMS
)
    compute_equilibrium(policy, params)
end

export compute_equilibrium, TechnologyDistribution
