using JuMP
using Ipopt
using Statistics
using Distributions
using QuadGK  # For numerical integration

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
        η_mean::Float64         # Mean technology level
        η_std::Float64         # Technology dispersion
        credibility::Float64    # Policy maker credibility
    end

    function form_tax_expectations(
        current_tax::Float64,
        announced_tax::Float64,
        η_mean::Float64,
        η_std::Float64,
        credibility::Float64
    )
        return PolicyExpectations(
            current_tax,
            announced_tax,
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
function sample_technology(params::ModelParameters)
    # Define the covariance matrix
    Σ = [params.σ_A^2 params.ρ * params.σ_A * params.σ_η;
         params.ρ * params.σ_A * params.σ_η params.σ_η^2]
    
    μ = [params.μ_A, params.μ_η]
    dist = MvNormal(μ, Σ)
    
    # Sample period 0 technology
    samples_0 = rand(dist, 1)
    A_0, η_0 = samples_0[1, 1], samples_0[2, 1]
    
    # Sample multiple potential period 1 technologies
    n_samples = 5  # Number of potential future technologies to consider
    potential_techs = rand(dist, n_samples)
    
    # Calculate effective productivity for each potential technology
    # Using a reference tax rate (could be made parameter)
    τ_ref = 0.05  
    A_eff_0 = A_0 * (1 - τ_ref * η_0)
    
    best_tech = nothing
    best_improvement = 0.0
    
    for i in 1:n_samples
        A_1, η_1 = potential_techs[1, i], potential_techs[2, i]
        A_eff_1 = A_1 * (1 - τ_ref * η_1)
        
        # Calculate relative improvement
        improvement = (A_eff_1 - A_eff_0) / A_eff_0
        
        if improvement > best_improvement
            best_improvement = improvement
            best_tech = (A_1, η_1)
        end
    end
    
    # If no improvement found, stay with current technology
    if best_tech === nothing || best_improvement < 0.01  # Minimum threshold for change
        return (A_0, η_0, A_0, η_0)  # No technology change
    else
        return (A_0, η_0, best_tech[1], best_tech[2])
    end
end

# Core equilibrium solver with partial investment
function compute_equilibrium_core(τ::Float64, A_0::Float64, η_0::Float64, A_1::Float64, η_1::Float64, skill_factor::Float64, params::ModelParameters)
    # Normalize productivity relative to A_0
    normalization_factor = 1.0 / A_0
    A_0_norm = 1.0
    A_1_norm = A_1 * normalization_factor
    K_0 = params.K_init 
    
    potential_Y0 = (1 - τ*η_0) * (K_0^params.α)
    max_Y = 5.0 * potential_Y0
    
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_silent(model)
    
    # Core variables with investment fraction
    @variables(model, begin
        0.01 <= C_0 <= 5.0 * max_Y
        0.01 <= C_1 <= 5.0 * max_Y
        0.01 <= L_0 <= 1.0
        0.01 <= L_1 <= 1.0
        0.01 <= K_1 <= 3.0 * max_Y
        0.0 <= frac <= 1.0  # Fraction of capital using new technology
    end)

    # Compute effective technology parameters for period 1
    @expression(model, A_1_eff, (1-frac)*A_0_norm + frac*A_1_norm)
    @expression(model, η_1_eff, (1-frac)*η_0 + frac*η_1)
    
    # Modified labor efficiency calculation to work with JuMP expressions
    @expression(model, Δη, η_1_eff - η_0)
    @expression(model, labor_efficiency, 1 / (1 + params.γ * Δη * Δη * skill_factor))
    
    # Production with effective technology parameters
    @expression(model, Y_0, (1 - τ*η_0) * K_0^params.α * (labor_efficiency * L_0)^(1-params.α))
    @expression(model, Y_1, (1 - τ*η_1_eff) * A_1_eff * K_1^params.α * L_1^(1-params.α))
    
    # Modified wage to reflect effective labor and technology
    @expression(model, w_0, (1-params.α) * (1 - τ*η_0) * K_0^params.α * 
               (labor_efficiency * L_0)^(-params.α) * labor_efficiency)
    @expression(model, w_1, (1-params.α) * (1 - τ*η_1_eff) * A_1_eff * 
               K_1^params.α * L_1^(-params.α))

    # Standard constraints
    @expression(model, r_1, params.α * Y_1 / K_1 - params.δ)
    @constraint(model, C_0 + K_1 == Y_0 + (1-params.δ)*K_0)
    @constraint(model, C_1 == Y_1)
    @constraint(model, C_0^(-params.σ) == params.β * (1 + r_1) * C_1^(-params.σ))
    @constraint(model, params.χ * L_0^params.ν == w_0 * C_0^(-params.σ))
    @constraint(model, params.χ * L_1^params.ν == w_1 * C_1^(-params.σ))

    # Same objective function
    @objective(model, Max, 
        (C_0^(1-params.σ))/(1-params.σ) - params.χ*L_0^(1+params.ν)/(1+params.ν) +
        params.β * ((C_1^(1-params.σ))/(1-params.σ) - params.χ*L_1^(1+params.ν)/(1+params.ν))
    )

    optimize!(model)
    
    status = termination_status(model)
    if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
        error("Solver did not find an optimal solution. Status: $status")
    end
    
    # Return results with technology split information and evaluated labor efficiency
    return Dict(
        "C_0" => value(C_0) / normalization_factor,
        "C_1" => value(C_1) / normalization_factor,
        "L_0" => value(L_0),
        "L_1" => value(L_1),
        "K_1" => value(K_1) / normalization_factor,
        "w_0" => value(w_0) / normalization_factor,
        "w_1" => value(w_1) / normalization_factor,
        "r_1" => value(r_1),
        "Y_0" => value(Y_0) / normalization_factor,
        "Y_1" => value(Y_1) / normalization_factor,
        "A_0" => A_0,
        "A_1" => A_1,
        "η_0" => η_0,
        "η_1" => η_1,
        "Technology_Split" => value(frac),
        "Labor_Efficiency" => value(labor_efficiency),  # Evaluate the expression
        "Skill_Factor" => skill_factor,
        # Add these for more detailed analysis
        "Δη" => value(Δη),
        "A_1_eff" => value(A_1_eff),
        "η_1_eff" => value(η_1_eff)
    )
end

# Wrapper handles all stochastic elements
function compute_equilibrium(expectations::PolicyExpectations, params::ModelParameters = DEFAULT_PARAMS)
    # Pre-compute all stochastic elements
    A_0, η_0, A_1, η_1 = sample_technology(params)
    skill_factor = compute_aggregate_skill(params)
    
    τ_0 = expectations.τ_current
    
    # Call the deterministic core function
    result = try
        compute_equilibrium_core(τ_0, A_0, η_0, A_1, η_1, skill_factor, params)
    catch e
        println("Error in compute_equilibrium_core: $e")
        return Dict()
    end
    
    # Transform result to match expected API
    return Dict(
        "w_t" => result["w_0"],     
        "r_t" => result["r_1"],     
        "A_t" => result["A_0"],     
        "η_t" => result["η_0"],     
        "K_t" => result["K_1"],     # Note: Using K_1 instead of K_0
        "L_t" => result["L_0"],     
        "Y_t" => result["Y_0"],     
        "C_0" => result["C_0"],
        "Labor_Efficiency" => result["Labor_Efficiency"],
        "Technology_Split" => result["Technology_Split"]  # New field
    )
end

# Main function for direct execution
function main()
    params = DEFAULT_PARAMS
    results = compute_equilibrium(0.0, params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

