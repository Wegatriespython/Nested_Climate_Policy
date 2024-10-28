using JuMP
using Ipopt
using Statistics
using Distributions  # For joint distribution sampling

# Model Parameters Module
module ModelParametersModule
    export ModelParameters, DEFAULT_PARAMS

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
    end

    const DEFAULT_PARAMS = ModelParameters()
end

using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

# Function to sample technology parameters
function sample_technology(params::ModelParameters)
    # Define the covariance matrix
    Σ = [params.σ_A^2 params.ρ * params.σ_A * params.σ_η;
         params.ρ * params.σ_A * params.σ_η params.σ_η^2]
    
    μ = [params.μ_A, params.μ_η]
    dist = MvNormal(μ, Σ)
    
    # Sample and ensure period 1 is at least as good as period 0
    samples_0 = rand(dist, 1)
    A_0, η_0 = samples_0[1, 1], samples_0[2, 1]
    
    # Keep sampling for period 1 until we get better technology
    while true
        samples_1 = rand(dist, 1)
        A_1, η_1 = samples_1[1, 1], samples_1[2, 1]
        
        # Check if period 1 technology is better (higher effective productivity)
        if A_1 ≥ A_0 && η_1 ≤ η_0
            return (A_0, η_0, A_1, η_1)
        end
    end
end

# Main Equilibrium Solver
function compute_equilibrium(τ::Float64, params::ModelParameters = DEFAULT_PARAMS)
    # Sample technology parameters first
    A_0, η_0, A_1, η_1 = sample_technology(params)
    
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    # Variables (with tighter bounds for numerical stability)
    @variables(model, begin
        0.1 <= C_0 <= 2.0    # Consumption
        0.1 <= C_1 <= 2.0
        0.1 <= L_0 <= 1.0    # Labor (normalized to max of 1)
        0.1 <= L_1 <= 1.0
        0.1 <= K_1 <= 2.0    # Capital stock
        0.1 <= w_0 <= 2.0    # Wage
        0.1 <= w_1 <= 2.0
        0.0 <= r_1 <= 0.5    # Interest rate
    end)

    # Production and capital
    K_0 = params.K_init
    @expression(model, Y_0, (1 - τ*η_0)*A_0 * K_0^params.α * L_0^(1-params.α))
    @expression(model, Y_1, (1 - τ*η_1)*A_1 * K_1^params.α * L_1^(1-params.α))

    # Tax revenue
    @expression(model, Tax_0, τ*η_0*A_0 * K_0^params.α * L_0^(1-params.α))
    @expression(model, Tax_1, τ*η_1*A_1 * K_1^params.α * L_1^(1-params.α))

    # Market clearing and optimality conditions
    @constraint(model, w_0 == (1-params.α) * Y_0 / L_0)
    @constraint(model, w_1 == (1-params.α) * Y_1 / L_1)
    @constraint(model, r_1 == params.α * Y_1 / K_1 - params.δ)
    @constraint(model, C_0 + K_1 == Y_0 + (1-params.δ)*K_0 + Tax_0)
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
        error("Solver did not find an optimal solution. Status: $status")
    end

    return Dict(
        "C_0" => value(C_0),
        "C_1" => value(C_1),
        "L_0" => value(L_0),
        "L_1" => value(L_1),
        "K_0" => K_0,
        "K_1" => value(K_1),
        "w_0" => value(w_0),
        "w_1" => value(w_1),
        "r_1" => value(r_1),
        "Y_0" => value(Y_0),
        "Y_1" => value(Y_1),
        "Tax_0" => value(Tax_0),
        "Tax_1" => value(Tax_1),
        "A_0" => A_0,
        "A_1" => A_1,
        "η_0" => η_0,
        "η_1" => η_1
    )
end

# Example usage
params = DEFAULT_PARAMS
results = compute_equilibrium(0.0, params)

# Display results
println("Equilibrium Results:")
for (key, value) in results
    println("$key: $value")
end
