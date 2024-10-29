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

# Core equilibrium solver becomes purely deterministic
function compute_equilibrium_core(τ::Float64, A_0::Float64, η_0::Float64, A_1::Float64, η_1::Float64, skill_factor::Float64, params::ModelParameters)
    # Normalize productivity relative to A_0
    normalization_factor = 1.0 / A_0
    A_0_norm = 1.0  # A_0 * normalization_factor = 1.0
    A_1_norm = A_1 * normalization_factor
    
    # Scale initial capital and other parameters accordingly
    K_0_norm = params.K_init * normalization_factor
    
    # Estimate potential output with normalized values
    potential_Y0 = (1 - τ*η_0) * (K_0_norm^params.α)  # Now with A_0 = 1
    max_Y = 5.0 * potential_Y0
    
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)  # Suppress Ipopt output
    set_silent(model)  # Suppress JuMP output
    
    # Variables with adjusted bounds
    @variables(model, begin
        0.01 <= C_0 <= 5.0 * max_Y    # Allow for higher consumption
        0.01 <= C_1 <= 5.0 * max_Y
        0.01 <= L_0 <= 1.0    # Labor still normalized to 1 as it represents fraction of time
        0.01 <= L_1 <= 1.0
        0.01 <= K_1 <= 3.0 * max_Y  # Allow for higher capital accumulation
        0.01 <= w_0 <= 3.0 * max_Y  # Higher wage bounds
        0.01 <= w_1 <= 3.0 * max_Y
        0.0 <= r_1 <= 2.0    # Allow for higher returns
    end)

    # Compute effective labor adjustment from technology change
    Δη = η_1 - η_0
    labor_efficiency = if abs(Δη) < 1e-10
        1.0
    else
        1.0 / (1.0 + params.γ * (Δη^2) * skill_factor)  # Hyperbolic form ensures efficiency ∈ (0,1]
    end
    
    # Modified production function with labor efficiency
    @expression(model, Y_0, (1 - τ*η_0) * K_0_norm^params.α * (labor_efficiency * L_0)^(1-params.α))
    @expression(model, Y_1, (1 - τ*η_1) * A_1_norm * K_1^params.α * L_1^(1-params.α))

    # Tax revenue
    @expression(model, Tax_0, τ*η_0 * K_0_norm^params.α * (labor_efficiency * L_0)^(1-params.α))
    @expression(model, Tax_1, τ*η_1 * A_1_norm * K_1^params.α * L_1^(1-params.α))

    # Modified wage to reflect effective labor
    @constraint(model, w_0 == (1-params.α) * (1 - τ*η_0) * K_0_norm^params.α * 
               (labor_efficiency * L_0)^(-params.α) * labor_efficiency)
    @constraint(model, w_1 == (1-params.α) * (1 - τ*η_1) * A_1_norm * K_1^params.α * L_1^(-params.α))
    @constraint(model, r_1 == params.α * Y_1 / K_1 - params.δ)
    @constraint(model, C_0 + K_1 == Y_0 + (1-params.δ)*K_0_norm + Tax_0)
    @constraint(model, C_1 == Y_1 + Tax_1)
    @constraint(model, C_0^(-params.σ) == params.β * (1 + r_1) * C_1^(-params.σ))
    @constraint(model, params.χ * L_0^params.ν == w_0 * C_0^(-params.σ))
    @constraint(model, params.χ * L_1^params.ν == w_1 * C_1^(-params.σ))

    # New objective: maximize utility
    @objective(model, Max, 
        (C_0^(1-params.σ))/(1-params.σ) - params.χ*L_0^(1+params.ν)/(1+params.ν) +
        params.β * ((C_1^(1-params.σ))/(1-params.σ) - params.χ*L_1^(1+params.ν)/(1+params.ν))
    )

    optimize!(model)
    
    status = termination_status(model)
    if status != MOI.LOCALLY_SOLVED && status != MOI.OPTIMAL
        println("Skill factor: $skill_factor")
        println("Running the new model core")
        println("\nOptimization failed:")
        println("τ: $τ")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("labor_efficiency: $labor_efficiency")
        println("Status: $status")
        error("Solver did not find an optimal solution. Status: $status")
    end
    
    # De-normalize results before returning
    if status in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        return Dict(
            "C_0" => value(C_0) / normalization_factor,
            "C_1" => value(C_1) / normalization_factor,
            "L_0" => value(L_0),  # Labor remains normalized
            "L_1" => value(L_1),
            "K_0" => K_0_norm / normalization_factor,
            "K_1" => value(K_1) / normalization_factor,
            "w_0" => value(w_0) / normalization_factor,
            "w_1" => value(w_1) / normalization_factor,
            "r_1" => value(r_1),  # Returns are already normalized
            "Y_0" => value(Y_0) / normalization_factor,
            "Y_1" => value(Y_1) / normalization_factor,
            "Tax_0" => value(Tax_0) / normalization_factor,
            "Tax_1" => value(Tax_1) / normalization_factor,
            "A_0" => A_0,  # Keep original values for reference
            "A_1" => A_1,
            "η_0" => η_0,
            "η_1" => η_1,
            "Labor_Efficiency" => labor_efficiency,
            "Skill_Factor" => skill_factor
        )
    else
        error("Solver did not find an optimal solution. Status: $(termination_status(model))")
    end
end

# Wrapper handles all stochastic elements
function compute_equilibrium(expectations::PolicyExpectations, params::ModelParameters = DEFAULT_PARAMS)
    # Pre-compute all stochastic elements
    A_0, η_0, A_1, η_1 = sample_technology(params)
    skill_factor = compute_aggregate_skill(params)
    
    τ_0 = expectations.τ_current
    τ_1 = expectations.τ_announced * expectations.credibility + 
          expectations.τ_current * (1 - expectations.credibility)
    
    # Call the deterministic core function
    result = try
        compute_equilibrium_core(τ_0, A_0, η_0, A_1, η_1, skill_factor, params)
    catch e
        println("Error in compute_equilibrium_core: $e")
        println("τ_0: $τ_0, A_0: $A_0, η_0: $η_0, A_1: $A_1, η_1: $η_1, skill_factor: $skill_factor, params: $params")
        return Dict()
    end
    
    # Transform result
    return Dict(
        "w_t" => result["w_0"],     
        "r_t" => result["r_1"],     
        "A_t" => result["A_0"],     
        "η_t" => result["η_0"],     
        "K_t" => result["K_0"],     
        "L_t" => result["L_0"],     
        "Y_t" => result["Y_0"],     
        "C_0" => result["C_0"]      
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

