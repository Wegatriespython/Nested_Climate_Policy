using NLsolve
include("economic_functions.jl")

function compute_equilibrium(tax_expectations::PolicyExpectations, 
                           params::ModelParameters = DEFAULT_PARAMS)
    prices_guess = [1.0, 0.05]
    
    function equilibrium_conditions(prices)
        w_t, r_t = prices
        
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

function form_tax_expectations(τ_current::Float64, τ_announced::Float64, 
                             η_mean::Float64, η_std::Float64, credibility::Float64)
    return PolicyExpectations(τ_current, τ_announced, η_mean, η_std, credibility)
end

export compute_equilibrium, form_tax_expectations
