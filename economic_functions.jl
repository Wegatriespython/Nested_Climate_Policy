using Optim
include("types.jl")

# Core economic functions
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
            K_L_ratio = (params.α / (1 - params.α)) * (w_t / r_t)
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

function sample_A_eta(n, tech_params::Dict{String, Float64})
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

export adjustment_cost, optimal_technology_choice, labor_supply, firm_decision, sample_A_eta
