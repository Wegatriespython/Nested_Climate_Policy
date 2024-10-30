using JuMP
using Ipopt
using Statistics
using Distributions
using QuadGK  # For numerical integration
using Random

# Model Parameters Module
module ModelParametersModule
    export ModelParameters, DEFAULT_PARAMS, PolicyExpectations, form_tax_expectations

    Base.@kwdef struct ModelParameters
        # Standard macro parameters
        β::Float64 = 0.96          # Discount factor
        σ::Float64 = 2.0           # Relative risk aversion
        χ::Float64 = 1.0           # Labor disutility
        ν::Float64 = 1.0           # Inverse Frisch elasticity
        α::Float64 = 0.33          # Capital share
        δ::Float64 = 0.1           # Depreciation rate
        K_init::Float64 = 1.0      # Initial capital
        
        # Technology distribution parameters
        μ_A::Float64 = 1.0         # Mean productivity
        μ_η::Float64 = 1.0         # Mean carbon intensity
        σ_A::Float64 = 0.2         # Std dev of productivity
        σ_η::Float64 = 0.2         # Std dev of carbon intensity
        ρ::Float64 = 0.5           # Correlation coefficient
        
        # Skill distribution parameters
        θ_min::Float64 = 0.1       # Minimum skill level
        θ_max::Float64 = 1.0       # Maximum skill level
        γ::Float64 = 0.1          # Reduced from 0.5 to account for output scaling
        
        # Partial investment parameters
        investment_adjustment_cost::Float64 = 0.1  # Cost of adjusting investment allocation
        min_investment_fraction::Float64 = 0.0     # Minimum fraction for new technology
        max_investment_fraction::Float64 = 1.0     # Maximum fraction for new technology
    end

    struct PolicyExpectations
        τ_current::Float64      # Current tax rate
        τ_announced::Float64    # Announced future tax rate
        A_mean::Float64         # Mean technology level
        A_std::Float64         # Technology dispersion
        η_mean::Float64         # Mean carbon intensity
        η_std::Float64         # Carbon intensity dispersion
        credibility::Float64    # Policy maker credibility
    end

    function form_tax_expectations(
        current_tax::Float64,
        announced_tax::Float64,
        A_mean :: Float64,
        A_std :: Float64,
        η_mean::Float64,
        η_std::Float64,
        credibility::Float64
    )

        println("forming tax expectations: η_mean: $η_mean, η_std: $η_std")
        return PolicyExpectations(
            current_tax,
            announced_tax,
            A_mean,
            A_std,
            η_mean,
            η_std,
            credibility
        )
    end

    const DEFAULT_PARAMS = ModelParameters()
end

using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS, PolicyExpectations, form_tax_expectations

# Pre-compute aggregate skill factor
function compute_aggregate_skill(params::ModelParameters)
    # Uniform distribution of skills for simplicity
    # Integrate 1/θ over the skill distribution
    skill_factor, _ = quadgk(θ -> 1/θ, params.θ_min, params.θ_max)
    return skill_factor / (params.θ_max - params.θ_min)  # Average effect
end

# Compute adjustment cost given change in carbon intensity
function compute_adjustment_cost(Δη::Float64, skill_factor::Float64, Y::Float64, params::ModelParameters)
    if abs(Δη) < 1e-10  # No technology change
        return 0.0
    else
        # Scale adjustment cost relative to output
        return params.γ * (Δη^2) * skill_factor * Y
    end
end

# Function to sample technology parameters
function sample_technology(expectations::PolicyExpectations, params::ModelParameters)
    # Define the covariance matrix
    A_mean = expectations.A_mean
    A_std = expectations.A_std
    η_mean = expectations.η_mean
    η_std = expectations.η_std
    τ_0 = expectations.τ_current  
    τ_1 = expectations.τ_announced

    # Safety bounds for η based on tax rates
    η_max_0 = τ_0 > 0 ? (0.99/τ_0) * 0.95 : Inf  # Avoid exactly 1/τ to prevent division by zero
    η_max_1 = τ_1 > 0 ? (0.99/τ_1) * 0.95 : Inf

    Σ = [A_std^2              params.ρ * A_std * η_std;
         params.ρ * A_std * η_std    η_std^2]
    
    μ = [A_mean, η_mean]
    dist = MvNormal(μ, Σ)
    
    # Sample period 0 technology with η constraint and non-negativity
    local A_0, η_0
    while true
        samples = rand(dist, 1)
        A_0, η_0 = samples[1, 1], samples[2, 1]
        if η_0 < η_max_0 && η_0 > 0 && A_0 > 0
            break
        end
    end
    
    # Sample multiple potential period 1 technologies
    n_samples = 100
    best_tech = nothing
    best_improvement = 0.0
    
    A_eff_0 = A_0 * (1 - τ_0 * η_0)
    
    # Sample and check period 1 technologies
    for _ in 1:n_samples
        samples = rand(dist, 1)
        A_1, η_1 = samples[1, 1], samples[2, 1]
        
        # Skip if η_1 exceeds bounds or if either value is negative
        if η_1 >= η_max_1 || η_1 <= 0 || A_1 <= 0
            continue
        end
        
        A_eff_1 = A_1 * (1 - τ_1 * η_1)
        improvement = (A_eff_1 - A_eff_0) / A_eff_0
        
        if improvement > best_improvement
            best_improvement = improvement
            best_tech = (A_1, η_1)
        end
    end
    
    # If no valid improvement found, stay with current technology
    if best_tech === nothing || best_improvement < 0.01
        println("No valid improvement found, staying with current technology")
        return (A_0, η_0, A_0, η_0)
    else
        return (A_0, η_0, best_tech[1], best_tech[2])
    end
end

# Core equilibrium solver with partial investment
function compute_equilibrium_core(τ_0::Float64, τ_1::Float64, A_init::Float64, η_init::Float64, A_0::Float64, η_0::Float64, A_1::Float64, η_1::Float64, skill_factor::Float64, params::ModelParameters)
    # Initial conditions and normalizations
    K_init = params.K_init
    
    # Base output for scaling (using initial technology)
    Y_base = A_init * (K_init^params.α)
    output_norm = 1.0 / Y_base

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    # Adjust solver parameters for better numerical stability
    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "acceptable_tol", 1e-4)

    #set model to silent

    
    # Variables with starting points - corrected syntax
    @variable(model, C_0 >= 0.001, start = 0.5)
    @variable(model, C_1 >= 0.001, start = 0.5)
    @variable(model, 0.001 <= L_0 <= 1.0, start = 0.3)
    @variable(model, 0.001 <= L_1 <= 1.0, start = 0.3)
    @variable(model, I_0 >= 0.001, start = K_init)
    @variable(model, I_1 >= 0.001, start = K_init)

    # Set upper bounds separately if needed
    set_upper_bound(C_0, 5.0)
    set_upper_bound(C_1, 5.0)

    
    # Capital stock expressions
    @expression(model, K_stock_0, K_init + I_0)
    @expression(model, K_stock_1, (1-params.δ)*K_stock_0 + I_1)
    
    # First define the investment fractions
    @expression(model, frac_0, I_0/K_init)
    @expression(model, frac_1, I_1/K_stock_0)  


                    
    # Technology adoption expressions with safeguards (now frac_0 and frac_1 are defined)
    @expression(model, A_eff_0, ((1-frac_0) * A_init + frac_0* A_0))             
    @expression(model, η_eff_0, ((1-frac_0) * η_init + frac_0 * η_0))
    @expression(model, A_eff_1, ((1-frac_1) * A_eff_0 + frac_1 * A_1))
    @expression(model, η_eff_1, ((1-frac_1) * η_eff_0 + frac_1 * η_1))
    
    # Adjustment costs with numerical safeguards
    @expression(model, Δη_0, η_eff_0 - η_init)
    @expression(model, Δη_1, η_eff_1 - η_eff_0)
    @expression(model, labor_efficiency_0, 1 / (1 + params.γ * Δη_0^2 * skill_factor))
    @expression(model, labor_efficiency_1, 1 / (1 + params.γ * Δη_1^2 * skill_factor))
    
    # Production with safeguards
    @expression(model, Y_0, (1 - τ_0*η_eff_0) * A_eff_0 * K_stock_0^params.α * 
               (labor_efficiency_0 * L_0)^(1-params.α))
    @expression(model, Y_1, (1 - τ_1*η_eff_1) * A_eff_1 * K_stock_1^params.α * 
               (labor_efficiency_1 * L_1)^(1-params.α))
    
    # Interest rates with safeguards
    @expression(model, r_0, params.α * Y_0 / max(K_stock_0, 0.001) - params.δ)
    @expression(model, r_1, params.α * Y_1 / max(K_stock_1, 0.001) - params.δ)
    
    # Wages
    @expression(model, w_0, (1-params.α) * Y_0 / max(labor_efficiency_0 * L_0, 0.001))
    @expression(model, w_1, (1-params.α) * Y_1 / max(labor_efficiency_1 * L_1, 0.001))

    # Budget constraints
    @constraint(model, C_0 + I_0 == Y_0)
    @constraint(model, C_1 + I_1 == Y_1)

    # Euler equations (reformulated for better numerical stability)
    @constraint(model, params.σ * (log(C_1) - log(C_0)) == log(params.β) + log(1 + r_1))

    @constraint(model, log(params.χ) + params.ν*log(L_0) == log(w_0) - params.σ*log(C_0))
    @constraint(model, log(params.χ) + params.ν*log(L_1) == log(w_1) - params.σ*log(C_1))

    @objective(model, Max, 
        (C_0^(1-params.σ))/(1-params.σ) - params.χ*L_0^(1+params.ν)/(1+params.ν) +
        params.β * ((C_1^(1-params.σ))/(1-params.σ) - params.χ*L_1^(1+params.ν)/(1+params.ν))
    )

    # Add these constraints after all expressions are defined
    @constraint(model, 0 <= frac_0 <= 1.0)
    @constraint(model, 0 <= frac_1 <= 1.0)

    optimize!(model)
    
    status = termination_status(model)
    if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
        # Create detailed parameter dump
        param_dump = """
        Optimization failed with status: $status
        
        Input Parameters:
        ----------------
        τ_0: $τ_0
        τ_1: $τ_1
        A_init: $A_init
        η_init: $η_init
        A_0: $A_0, η_0: $η_0
        A_1: $A_1, η_1: $η_1
        skill_factor: $skill_factor
        
        K_init: $(params.K_init)
        
        Remaining Parameters:
        --------------------
        $(params)
        """
        println(param_dump)
        error(param_dump)
    end
    
    # Return results with technology split information and evaluated labor efficiency
    return Dict(
        "C_0" => value(C_0) / output_norm,
        "C_1" => value(C_1) / output_norm,
        "L_0" => value(L_0),
        "L_1" => value(L_1),
        "K_0" => value(K_stock_0) / output_norm,
        "K_1" => value(K_stock_1) / output_norm,
        "w_0" => value(w_0) / output_norm,
        "w_1" => value(w_1) / output_norm,
        "r_0" => value(r_0),
        "r_1" => value(r_1),
        "Y_0" => value(Y_0) / output_norm,
        "Y_1" => value(Y_1) / output_norm,
        "E_0" => value(Y_0 * η_eff_0),
        "E_1" => value(Y_1 * η_eff_1),
        "A_0" => A_0,
        "A_1" => A_1,
        "η_0" => η_0,
        "η_1" => η_1,
        "Technology_Split_0" => value(frac_0),
        "Technology_Split_1" => value(frac_1),
        "Labor_Efficiency_0" => value(labor_efficiency_0),
        "Labor_Efficiency_1" => value(labor_efficiency_1),
        "A_0_eff" => value(A_eff_0),
        "A_1_eff" => value(A_eff_1),
        "η_0_eff" => value(η_eff_0),
        "η_1_eff" => value(η_eff_1)
    )
end

# Wrapper handles all stochastic elements
function compute_equilibrium(expectations::PolicyExpectations, params::ModelParameters = DEFAULT_PARAMS)
    # Pre-compute all stochastic elements
    #Set random seed

    A_0, η_0, A_1, η_1 = sample_technology(expectations, params)
    A_init, η_init = A_0*0.95, η_0*1.05
    skill_factor = compute_aggregate_skill(params)
    
    τ_0 = expectations.τ_current
    τ_1 = expectations.τ_announced *expectations.credibility
    
    # Call the deterministic core function
    result = try
        compute_equilibrium_core(τ_0,τ_1, A_init, η_init, A_0, η_0, A_1, η_1, skill_factor, params)
    catch e
        return Dict(
            "error" => true,
            "error_message" => string(e)
        )
    end
    
    if get(result, "error", false)
        return result
    end
    
    # Transform result to match expected API
    return Dict(
        "w_0" => result["w_0"],
        "w_1" => result["w_1"],
        "r_0" => result["r_0"],
        "r_1" => result["r_1"],
        "A_0" => result["A_0"],
        "A_1" => result["A_1"],
        "η_0" => result["η_0"],
        "η_1" => result["η_1"],
        "K_0" => params.K_init,
        "K_1" => result["K_1"],
        "L_0" => result["L_0"],
        "L_1" => result["L_1"],
        "Y_0" => result["Y_0"],
        "Y_1" => result["Y_1"],
        "C_0" => result["C_0"],
        "C_1" => result["C_1"],
        "E_0" => result["E_0"],
        "E_1" => result["E_1"],
        "Labor_Efficiency_0" => result["Labor_Efficiency_0"],
        "Labor_Efficiency_1" => result["Labor_Efficiency_1"],
        "Technology_Split_0" => result["Technology_Split_0"],
        "Technology_Split_1" => result["Technology_Split_1"]
    )
end

# Main function for direct execution
function main()
    Random.seed!(12344)
    params = DEFAULT_PARAMS
    #Parameters for policy expectations: current tax rate, announced tax rate, mean technology level, technology dispersion, mean carbon intensity, carbon intensity dispersion, credibility
    #form_tax_expectations(τ_0, τ_1, A_mean, A_std, η_mean, η_std, credibility)
    policy_expectations = form_tax_expectations(0.1, 0.1, 1.0, 0.05, 0.3, 0.01, 1.0)
    
    results = compute_equilibrium(policy_expectations, params)
    println(results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

