using JuMP
using Ipopt

# Model Parameters
struct ModelParameters
    α::Float64     # Capital share
    β::Float64     # Discount factor
    σ::Float64     # Relative risk aversion
    δ::Float64     # Depreciation rate
    χ::Float64     # Disutility of labor parameter
    ν::Float64     # Inverse of labor supply elasticity
    γ::Float64     # Adjustment cost parameter
    K_init::Float64  # Initial capital
end

# Core equilibrium solver function
function compute_equilibrium_core(
    τ_0::Float64, τ_1::Float64,          # Carbon taxes
    A_init::Float64, η_init::Float64,    # Initial technology and emissions intensity
    A_0::Float64, η_0::Float64,          # Technology and emissions intensity for investment in period 0
    A_1::Float64, η_1::Float64,          # Technology and emissions intensity for investment in period 1
    skill_factor::Float64,               # Skill factor for adjustment costs
    params::ModelParameters              # Model parameters
)
    # Initial conditions
    K_init = params.K_init

    model = Model(Ipopt.Optimizer)

    # Solver settings
    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-6)
    set_optimizer_attribute(model, "print_level", 0)

    # Variables
    @variable(model, C_0 >= 0.001, start = 1.0)
    @variable(model, C_1 >= 0.001, start = 1.0)
    @variable(model, 0.001 <= L_0 <= 1.0, start = 0.5)
    @variable(model, 0.001 <= L_1 <= 1.0, start = 0.5)
    @variable(model, K_0 >= 0.001, start = 0.5)
    @variable(model, K_1 >= 0.001, start = 0.5)
    @variable(model, 0 <= frac_1 <= 1.0, start = 0.5)

    # Capital stock in period 0
    @expression(model, K_stock_0, K_init + K_0)

    # Effective technology and emissions intensity in period 0
    @expression(model, A_eff_0, (K_init * A_init + K_0 * A_0) / K_stock_0)
    @expression(model, η_eff_0, (K_init * η_init + K_0 * η_0) / K_stock_0)

    # Capital stock in period 1
    @expression(model, K_stock_1, (1 - params.δ) * K_stock_0 + K_1)

    # Capital replacement in period 1
    @expression(model, K_old, (1 - frac_1) * (1 - params.δ) * K_stock_0)
    @expression(model, K_new, frac_1 * (1 - params.δ) * K_stock_0 + K_1)

    # Effective technology and emissions intensity in period 1
    @expression(model, A_eff_1, (K_old * A_eff_0 + K_new * A_1) / K_stock_1)
    @expression(model, η_eff_1, (K_old * η_eff_0 + K_new * η_1) / K_stock_1)

    # Change in emissions intensity
    @expression(model, Δη_0, η_eff_0 - η_init)
    @expression(model, Δη_1, η_eff_1 - η_eff_0)

    # Labor efficiency
    @expression(model, labor_efficiency_0, 1 / (1 + params.γ * Δη_0^2 * skill_factor))
    @expression(model, labor_efficiency_1, 1 / (1 + params.γ * Δη_1^2 * skill_factor))

    # Production functions
    @expression(model, Y_0, (1 - τ_0 * η_eff_0) * A_eff_0 * K_stock_0^params.α * (labor_efficiency_0 * L_0)^(1 - params.α))
    @expression(model, Y_1, (1 - τ_1 * η_eff_1) * A_eff_1 * K_stock_1^params.α * (labor_efficiency_1 * L_1)^(1 - params.α))

    # Interest rates
    @expression(model, r_0, params.α * Y_0 / K_stock_0 - params.δ)
    @expression(model, r_1, params.α * Y_1 / K_stock_1 - params.δ)

    # Wages
    @expression(model, w_0, (1 - params.α) * Y_0 / (labor_efficiency_0 * L_0))
    @expression(model, w_1, (1 - params.α) * Y_1 / (labor_efficiency_1 * L_1))

    # Budget constraints
    @constraint(model, C_0 + K_0 == Y_0)
    @constraint(model, C_1 + K_1 == Y_1)

    # Euler equation
    @constraint(model, params.σ * (log(C_1) - log(C_0)) == log(params.β) + log(1 + r_1))

    # Labor supply FOCs
    @constraint(model, log(params.χ) + params.ν * log(L_0) == log(w_0) - params.σ * log(C_0))
    @constraint(model, log(params.χ) + params.ν * log(L_1) == log(w_1) - params.σ * log(C_1))

    # Objective function
    @objective(model, Max,
        (C_0^(1 - params.σ)) / (1 - params.σ) - params.χ * L_0^(1 + params.ν) / (1 + params.ν) +
        params.β * ((C_1^(1 - params.σ)) / (1 - params.σ) - params.χ * L_1^(1 + params.ν) / (1 + params.ν))
    )

    # Solve the model
    optimize!(model)

    # Extract results
    solution = Dict(
        "C_0" => value(C_0),
        "C_1" => value(C_1),
        "L_0" => value(L_0),
        "L_1" => value(L_1),
        "K_0" => value(K_0),
        "K_1" => value(K_1),
        "frac_1" => value(frac_1),
        "A_eff_0" => value(A_eff_0),
        "A_eff_1" => value(A_eff_1),
        "η_eff_0" => value(η_eff_0),
        "η_eff_1" => value(η_eff_1),
        "Y_0" => value(Y_0),
        "Y_1" => value(Y_1),
        "r_0" => value(r_0),
        "r_1" => value(r_1),
        "w_0" => value(w_0),
        "w_1" => value(w_1)
    )

    return solution
end

# Example usage with dummy parameter values
params = ModelParameters(
    α = 0.3,
    β = 0.95,
    σ = 2.0,
    δ = 0.1,
    χ = 1.0,
    ν = 1.0,
    γ = 0.5,
    K_init = 1.0
)

# Call the function with example inputs
solution = compute_equilibrium_core(
    τ_0 = 0.1, τ_1 = 0.2,
    A_init = 1.0, η_init = 1.0,
    A_0 = 1.1, η_0 = 0.9,
    A_1 = 1.2, η_1 = 0.8,
    skill_factor = 1.0,
    params = params
)

# Print the solution
for (var, val) in solution
    println("$var = $val")
end
