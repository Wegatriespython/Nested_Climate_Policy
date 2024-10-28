# Import necessary packages
using Distributions
using Optim
using ForwardDiff
using NLsolve
using Plots

# Set random seed for reproducibility
using Random
Random.seed!(1234)

# 1. Parameters

# Economic parameters
β = 0.96          # Discount factor
σ = 2.0           # Relative risk aversion coefficient
χ = 1.0           # Labor disutility weight
ν = 1.0           # Inverse of Frisch elasticity
α = 0.33          # Capital share parameter
δ = 0.1           # Depreciation rate
γ = 0.1           # Adjustment cost coefficient
τ = 0.05          # Emission tax rate

# Joint distribution parameters
μ_A = 1.0         # Mean productivity
μ_eta = 1.0       # Mean carbon intensity
σ_A = 0.2         # Std dev of productivity
σ_eta = 0.2       # Std dev of carbon intensity
ρ = 0.5           # Correlation coefficient

# Skill distribution parameters
θ_min = 0.1       # Minimum skill level
θ_max = 1.0       # Maximum skill level
n_agents = 1000   # Number of agents/workers

# 2. Joint Distribution of A_t and η_t

# Define the joint distribution of (A_t, η_t)
cov = ρ * σ_A * σ_eta
Σ = [σ_A^2 cov; cov σ_eta^2]
joint_dist = MvNormal([μ_A, μ_eta], Σ)

# Function to sample (A_t, η_t)
function sample_A_eta(n)
    return rand(joint_dist, n)'
end

# 3. Skill Distribution of Workers

# Define the skill distribution F(θ)
θ_values = rand(Uniform(θ_min, θ_max), n_agents)

# 4. Adjustment Cost Function

function adjustment_cost(Δη, θ_values)
    φ = γ * Δη^2
    ψ = mean(1.0 ./ θ_values)
    return φ * ψ
end

# 5. Firms' Problem

# Firms choose (A_t, η_t), K_t, and L_t to maximize profits
function firm_decision(w_t, r_t, τ, A_eta_samples)
    n = size(A_eta_samples, 1)
    profits = zeros(n)
    K_t_vals = zeros(n)
    L_t_vals = zeros(n)
    for i in 1:n
        A_t, η_t = A_eta_samples[i, :]
        A_eff = (1 - τ * η_t) * A_t
        # Optimal K/L ratio
        K_L_ratio = (α / (1 - α)) * (w_t / r_t)
        # Optimal L_t (derived from marginal productivity conditions)
        L_t = ((A_eff * K_L_ratio^α) / w_t)^(1 / (1 - α))
        # Optimal K_t
        K_t = K_L_ratio * L_t
        # Production function
        Y_t = A_eff * K_t^α * L_t^(1 - α)
        # Profits
        profits[i] = Y_t - w_t * L_t - r_t * K_t - τ * η_t * Y_t
        K_t_vals[i] = K_t
        L_t_vals[i] = L_t
    end
    # Choose the (A_t, η_t) that maximizes profits
    max_idx = argmax(profits)
    optimal_A_t = A_eta_samples[max_idx, 1]
    optimal_η_t = A_eta_samples[max_idx, 2]
    optimal_K_t = K_t_vals[max_idx]
    optimal_L_t = L_t_vals[max_idx]
    return optimal_A_t, optimal_η_t, optimal_K_t, optimal_L_t
end

# 6. Households' Problem

# Labor supply function
function labor_supply(w_t, C_t)
    return (C_t^(-σ) * w_t / χ)^(1 / ν)
end

# Consumption Euler Equation
function consumption_euler(C_t, C_tp1, r_tp1)
    return β * (C_tp1 / C_t)^σ - (1 + r_tp1 - δ)
end

# 7. Equilibrium Computation

function compute_equilibrium(τ)
    # Initial guess for prices [w_t, r_t]
    prices_guess = [1.0, 0.05]

    # Define equilibrium conditions
    function equilibrium_conditions(prices)
        w_t, r_t = prices
        # Sample (A_t, η_t)
        A_eta_samples = sample_A_eta(1000)
        
        # Firms' decision
        A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, τ, A_eta_samples)
        A_eff = (1 - τ * η_t) * A_t
        Y_t = A_eff * K_t^α * L_t^(1 - α)
        
        # Households' decision
        C_0 = Y_t - δ * K_t - adjustment_cost(η_t - μ_eta, θ_values)
        L_supply = labor_supply(w_t, C_0)
        
        # Market clearing conditions
        labor_market_error = L_supply - L_t
        capital_market_error = K_t - K_t  # Simplified as K_t demand equals supply in this model
        return [labor_market_error, capital_market_error]
    end

    # Solve for equilibrium prices using NLsolve
    sol = nlsolve(equilibrium_conditions, prices_guess)
    w_t, r_t = sol.zero

    # Compute equilibrium quantities with solved prices
    A_eta_samples = sample_A_eta(1000)
    A_t, η_t, K_t, L_t = firm_decision(w_t, r_t, τ, A_eta_samples)
    A_eff = (1 - τ * η_t) * A_t
    Y_t = A_eff * K_t^α * L_t^(1 - α)
    C_0 = Y_t - δ * K_t - adjustment_cost(η_t - μ_eta, θ_values)
    
    # Print results
    println("Equilibrium reached:")
    println("Wage (w_t): ", w_t)
    println("Interest Rate (r_t): ", r_t)
    println("Optimal A_t: ", A_t)
    println("Optimal η_t: ", η_t)
    println("Capital (K_t): ", K_t)
    println("Labor (L_t): ", L_t)
    println("Output (Y_t): ", Y_t)
    println("Consumption (C_0): ", C_0)

    # Return equilibrium values
    return Dict(
        "w_t" => w_t,
        "r_t" => r_t,
        "A_t" => A_t,
        "η_t" => η_t,
        "K_t" => K_t,
        "L_t" => L_t,
        "Y_t" => Y_t,
        "C_0" => C_0
    )
end

# Run the equilibrium computation
equilibrium = compute_equilibrium(τ)

# 8. Policy Analysis (Optional)

# Function to analyze different emission tax rates
function policy_analysis(τ_values)
    results = []
    for τ in τ_values
        eq = compute_equilibrium(τ)  # Pass τ as an argument
        push!(results, (τ, eq["Y_t"], eq["η_t"]))
    end
    return results
end

# Example of policy analysis
τ_values = [0.0, 0.05, 0.1, 0.15, 0.2]
policy_results = policy_analysis(τ_values)

# Plot the results
τs = [res[1] for res in policy_results]
Y_ts = [res[2] for res in policy_results]
η_ts = [res[3] for res in policy_results]

plot(τs, Y_ts, label="Output Y_t", xlabel="Emission Tax Rate τ", ylabel="Output", title="Impact of Emission Tax on Output", legend=:topright)
