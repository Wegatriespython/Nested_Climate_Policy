using Distributions
using Optim
using LinearAlgebra

# Import our custom types and parameters
include("../types.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

"""
Computes adjustment cost based on skill distribution and change in technology
"""
function adjustment_cost(Δη::T, params::ModelParameters = DEFAULT_PARAMS) where {T<:Real}
    θ_values = get_skill_distribution(params)
    ψ = mean(1.0 ./ θ_values)
    return params.γ * Δη^2 * ψ
end

"""
Computes optimal technology choice given current state and expectations
"""
function optimal_technology_choice(
    current_η::Float64, 
    policy::PolicyExpectations,
    params::ModelParameters = DEFAULT_PARAMS
)
    function technology_cost(Δη)
        future_η = current_η + Δη
        tax_cost = policy.τ_announced * future_η
        adj_cost = adjustment_cost(Δη, params)
        return tax_cost + adj_cost
    end
    
    result = optimize(technology_cost, -0.5, 0.5)
    return result.minimizer
end

"""
Computes labor supply given wage and consumption
"""
function labor_supply(
    w_t::Float64, 
    C_t::Float64, 
    params::ModelParameters = DEFAULT_PARAMS
)
    return (C_t^(-params.σ) * w_t / params.χ)^(1 / params.ν)
end

"""
Samples from the joint distribution of productivity and carbon intensity
"""
function sample_technology_distribution(
    n::Int, 
    params::ModelParameters = DEFAULT_PARAMS
)
    # Create covariance matrix
    Σ = [params.σ_A^2 params.ρ*params.σ_A*params.σ_eta;
         params.ρ*params.σ_A*params.σ_eta params.σ_eta^2]
    
    # Create distribution
    dist = MvNormal([params.μ_A, params.μ_eta], Σ)
    
    return rand(dist, n)'
end

"""
Computes firm's optimal decisions given prices and expectations
"""
function firm_decision(
    w_t::Float64,
    r_t::Float64,
    policy::PolicyExpectations,
    params::ModelParameters = DEFAULT_PARAMS;
    n_samples::Int = 100
)
    # Sample technology pairs
    A_eta_samples = sample_technology_distribution(n_samples, params)
    
    # Initialize arrays for results
    profits = zeros(n_samples)
    K_t_vals = zeros(n_samples)
    L_t_vals = zeros(n_samples)
    
    for i in 1:n_samples
        A_t, η_t = A_eta_samples[i, :]
        A_eff = (1 - policy.τ_current * η_t) * A_t
        
        # Compute optimal technology change
        Δη = optimal_technology_choice(η_t, policy, params)
        
        # Compute optimal input ratios
        K_L_ratio = (params.α / (1 - params.α)) * (w_t / r_t)
        L_t = ((A_eff * K_L_ratio^params.α) / w_t)^(1 / (1 - params.α))
        K_t = K_L_ratio * L_t
        
        # Compute output and costs
        Y_t = A_eff * K_t^params.α * L_t^(1 - params.α)
        adj_costs = adjustment_cost(Δη, params)
        
        # Store results
        profits[i] = Y_t - w_t * L_t - r_t * K_t - 
                    policy.τ_current * η_t * Y_t - adj_costs
        K_t_vals[i] = K_t
        L_t_vals[i] = L_t
    end
    
    # Return optimal choice
    max_idx = argmax(profits)
    return (
        A = A_eta_samples[max_idx, 1],
        η = A_eta_samples[max_idx, 2],
        K = K_t_vals[max_idx],
        L = L_t_vals[max_idx]
    )
end

"""
Computes savings supply based on household optimization
"""
function savings_supply(
    r_t::Float64,
    w_t::Float64,
    C_t::Float64,
    params::ModelParameters = DEFAULT_PARAMS
)
    return (1 + r_t) * (w_t / (1 + params.δ)) * C_t^(-params.σ)
end

"""
Creates tax expectations structure
"""
function form_tax_expectations(
    τ_current::Float64,
    τ_announced::Float64,
    η_mean::Float64,
    η_std::Float64,
    credibility::Float64
)
    return PolicyExpectations(
        τ_current,
        τ_announced,
        η_mean,
        η_std,
        credibility
    )
end

export adjustment_cost, optimal_technology_choice, labor_supply,
       firm_decision, sample_technology_distribution, savings_supply,
       form_tax_expectations
