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
        σ_A::Float64 = 0.2         # Productivity standard deviation
        μ_η::Float64 = 1.0         # Mean emissions intensity
        σ_η::Float64 = 0.2         # Emissions intensity standard deviation
        ρ::Float64 = 0.5           # Correlation between A and η
        
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

# Current sampling function with suggested improvements
function sample_technology(expectations::PolicyExpectations, params::ModelParameters)
    # Define the joint distribution for productivity and emissions intensity
    A_mean = expectations.A_mean
    A_std = expectations.A_std
    η_mean = expectations.η_mean
    η_std = expectations.η_std
    
    # Create covariance matrix with correlation ρ
    Σ = [A_std^2              params.ρ * A_std * η_std;
         params.ρ * A_std * η_std    η_std^2]
    μ = [A_mean, η_mean]
    
    # Create truncated multivariate normal distribution
    # This ensures economically meaningful values
    function sample_truncated()
        while true
            samples = rand(MvNormal(μ, Σ), 1)
            A, η = samples[1, 1], samples[2, 1]
            
            # Economic constraints:
            # 1. Positive productivity
            # 2. Positive but bounded emissions intensity
            # 3. Ensure profitability condition: (1 - τη)A > 0
            if A > 0 && η > 0 && η < (0.99/max(expectations.τ_current, expectations.τ_announced))
                return (A, η)
            end
        end
    end
    
    # Sample initial period (-1) technology - no constraints
    n_samples = 10
    tech_minus1 = [sample_truncated() for _ in 1:n_samples]
    
    # Sample period 0 technologies with constraint relative to -1
    function sample_period_0()
        samples = [sample_truncated() for _ in 1:n_samples]
        # Filter for improvement over best period -1 technology
        base_productivity = maximum(A * (1 - expectations.τ_current * η) 
                                 for (A, η) in tech_minus1)
        return filter(tech -> 
            tech[1] * (1 - expectations.τ_current * tech[2]) > base_productivity,
            samples)
    end
    tech_0 = sample_period_0()
    
    # Sample period 1 technologies with constraint relative to 0
    function sample_period_1()
        samples = [sample_truncated() for _ in 1:n_samples]
        # Filter for improvement over best period 0 technology
        base_productivity = maximum(A * (1 - expectations.τ_announced * η) 
                                 for (A, η) in tech_0)
        return filter(tech -> 
            tech[1] * (1 - expectations.τ_announced * tech[2]) > base_productivity,
            samples)
    end
    tech_1 = sample_period_1()
    
    return (tech_minus1, tech_0, tech_1)
end

# Core equilibrium solver with partial investment
function compute_equilibrium_core(τ_0::Float64, τ_1::Float64, 
                                tech_minus1::Vector{Tuple{Float64,Float64}}, 
                                tech_0::Vector{Tuple{Float64,Float64}}, 
                                tech_1::Vector{Tuple{Float64,Float64}}, 
                                skill_factor::Float64, params::ModelParameters)
    K_init = params.K_init
    n_tech = length(tech_minus1)
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    # Portfolio variables
    @variable(model, K_0[1:n_tech] >= 0)  # Capital stock in each technology
    @variable(model, K_1[1:n_tech] >= 0)
    
    @variable(model, θ_0[1:n_tech] >= 0)  # Fraction of tech_minus1 replaced by tech_0
    @variable(model, θ_1[1:n_tech] >= 0)  # Fraction of portfolio_0 replaced by tech_1
    
    # Standard macro variables
    @variable(model, C_0 >= 0.001, start = 0.5)
    @variable(model, C_1 >= 0.001, start = 0.5)
    @variable(model, 0.001 <= L_0 <= 1.0, start = 0.3)
    @variable(model, 0.001 <= L_1 <= 1.0, start = 0.3)

    # Portfolio transitions
    @expression(model, portfolio_0[i=1:n_tech], 
        (1 - θ_0[i])*tech_minus1[i] + θ_0[i]*tech_0[i])
    
    @expression(model, portfolio_1[i=1:n_tech], 
        (1 - θ_1[i])*portfolio_0[i] + θ_1[i]*tech_1[i])

    # Effective technology levels
    @expression(model, A_eff_0, sum(
        (K_0[i]/K_total_0) * ((1 - θ_0[i])*tech_minus1[i][1] + θ_0[i]*tech_0[i][1])
        for i in 1:n_tech))
    
    @expression(model, η_eff_0, sum(
        (K_0[i]/K_total_0) * ((1 - θ_0[i])*tech_minus1[i][2] + θ_0[i]*tech_0[i][2])
        for i in 1:n_tech))
    
    @expression(model, A_eff_1, sum(
        (K_1[i]/K_total_1) * ((1 - θ_1[i])*portfolio_0[i][1] + θ_1[i]*tech_1[i][1])
        for i in 1:n_tech))
    
    @expression(model, η_eff_1, sum(
        (K_1[i]/K_total_1) * ((1 - θ_1[i])*portfolio_0[i][2] + θ_1[i]*tech_1[i][2])
        for i in 1:n_tech))

    # Technology change for adjustment costs
    @expression(model, Δη_0, η_eff_0 - sum((1/n_tech) * tech_minus1[i][2] for i in 1:n_tech))
    @expression(model, Δη_1, η_eff_1 - η_eff_0)

    # Capital evolution with portfolio transitions
    @constraint(model, [i=1:n_tech], 
        K_0[i] == (1-params.δ)*K_init/n_tech + 
                  (K_total_0 - (1-params.δ)*K_init)*θ_0[i])

    @constraint(model, [i=1:n_tech], 
        K_1[i] == (1-params.δ)*K_0[i] + 
                  (K_total_1 - (1-params.δ)*K_total_0)*θ_1[i])
    
    # Total capital expressions
    @expression(model, K_total_0, sum(K_0))
    @expression(model, K_total_1, sum(K_1))
    
    @constraint(model, [i=1:n_tech], 0 <= θ_0[i] <= 1)
    @constraint(model, [i=1:n_tech], 0 <= θ_1[i] <= 1)

    @constraint(model, sum(K_0[i]/K_total_θ for i in 1:n_tech) == 1)
    @constraint(model, sum(K_1[i]/K_total_1 for i in 1:n_tech) == 1)

    # Net investment calculations
    @expression(model, I_0, K_total_0 - (1-params.δ)*K_init)
    @expression(model, I_1, K_total_1 - (1-params.δ)*K_total_0)
    
    # Adjustment costs based on both rebalancing and capital changes
    @expression(model, tech_adjustment_0, 
        sum(θ_0[i]^2 * K_0[i]/K_total_0 for i in 1:n_tech))
    
    @expression(model, tech_adjustment_1,
        sum(θ_1[i]^2 * K_1[i]/K_total_1 for i in 1:n_tech))

    # Production function
    @expression(model, Y_0, (1 - τ_0*η_eff_0) * A_eff_0 * K_total_0^params.α * 
               (labor_efficiency_0 * L_0)^(1-params.α))
    @expression(model, Y_1, (1 - τ_1*η_eff_1) * A_eff_1 * K_total_1^params.α * 
               (labor_efficiency_1 * L_1)^(1-params.α))

    # Budget constraints include both investment and adjustment costs
    @constraint(model, C_0 + I_0 + params.investment_adjustment_cost * tech_adjustment_0 == Y_0)
    @constraint(model, C_1 + I_1 + params.investment_adjustment_cost * tech_adjustment_1 == Y_1)

    # Factor prices
    @expression(model, r_0, params.α * Y_0 / max(K_total_0, 0.001) - params.δ)
    @expression(model, r_1, params.α * Y_1 / max(K_total_1, 0.001) - params.δ)
    @expression(model, w_0, (1-params.α) * Y_0 / max(labor_efficiency_0 * L_0, 0.001))
    @expression(model, w_1, (1-params.α) * Y_1 / max(labor_efficiency_1 * L_1, 0.001))

    # Labor efficiency based on technology changes
    @expression(model, labor_efficiency_0, 1 / (1 + params.γ * Δη_0^2 * skill_factor))
    @expression(model, labor_efficiency_1, 1 / (1 + params.γ * Δη_1^2 * skill_factor))

    # Euler equations
    @constraint(model, params.σ * (log(C_1) - log(C_0)) == log(params.β) + log(1 + r_1))
    @constraint(model, log(params.χ) + params.ν*log(L_0) == log(w_0) - params.σ*log(C_0))
    @constraint(model, log(params.χ) + params.ν*log(L_1) == log(w_1) - params.σ*log(C_1))

    # Objective function
    @objective(model, Max, 
        (C_0^(1-params.σ))/(1-params.σ) - params.χ*L_0^(1+params.ν)/(1+params.ν) +
        params.β * ((C_1^(1-params.σ))/(1-params.σ) - params.χ*L_1^(1+params.ν)/(1+params.ν))
    )

    optimize!(model)
    
    # Error handling remains the same...
    status = termination_status(model)
    if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
        param_dump = """
        Optimization failed with status: $status
        
        Input Parameters:
        ----------------
        τ_0: $τ_0
        τ_1: $τ_1
        tech_minus1: $tech_minus1
        tech_0: $tech_0
        tech_1: $tech_1
        skill_factor: $skill_factor
        
        K_init: $(params.K_init)
        
        Remaining Parameters:
        --------------------
        $(params)
        """
        println(param_dump)
        error(param_dump)
    end

    # Return statement will need to be updated to include portfolio information...
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

