# Import necessary packages
using Distributions
using Optim
using ForwardDiff
using NLsolve
using Plots
using LinearAlgebra
using OptimBase: TwiceDifferentiable, TwiceDifferentiableConstraints

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

# Firm's optimization problem
function firm_optimize(w_t::Float64, r_t::Float64, τ::Float64, A_t::Float64, η_t::Float64)
    function firm_objective(x)
        K_t, L_t = x
        # Effective productivity accounting for emissions tax
        A_eff = (1 - τ * η_t) * A_t
        # Production
        Y_t = A_eff * K_t^α * L_t^(1 - α)
        # Profit = Revenue - Costs
        profit = Y_t - w_t * L_t - r_t * K_t
        return -profit  # Negative because Optim.jl minimizes
    end

    # Non-negativity constraints
    lower_bounds = [0.0, 0.0]
    upper_bounds = [Inf, Inf]
    initial_guess = [1.0, 1.0]
    
    # Create a TwiceDifferentiable object
    d = TwiceDifferentiable(firm_objective, initial_guess; autodiff=:forward)
    
    # Create box constraints
    db = TwiceDifferentiableConstraints(lower_bounds, upper_bounds)
    
    # Use IPNewton instead of Fminbox
    result = optimize(d, db, initial_guess, IPNewton())
    
    if !Optim.converged(result)
        @warn "Firm optimization did not converge"
    end
    
    K_t, L_t = Optim.minimizer(result)
    
    # Calculate resulting output and profit
    A_eff = (1 - τ * η_t) * A_t
    Y_t = A_eff * K_t^α * L_t^(1 - α)
    profit = -firm_objective([K_t, L_t])
    
    return K_t, L_t, Y_t, profit
end

# Household's optimization problem
function household_optimize(w_t::Float64, r_t::Float64, initial_assets::Float64)
    function household_utility(x)
        C_t, L_t = x
        # Standard utility function with consumption and labor disutility
        utility = C_t^(1-σ)/(1-σ) - χ * L_t^(1+ν)/(1+ν)
        return -utility  # Negative because Optim.jl minimizes
    end

    function budget_constraint(x)
        C_t, L_t = x
        # Budget constraint: consumption ≤ labor income + capital income
        return w_t * L_t + r_t * initial_assets - C_t
    end

    # Constraints
    lower_bounds = [0.0, 0.0]  # Non-negative consumption and labor
    upper_bounds = [Inf, 24.0]  # Max 24 hours of labor per day
    initial_guess = [1.0, 1.0]
    
    # Define constraint as dictionary for Optim
    constraints = TwiceDifferentiable(x -> -household_utility(x), initial_guess)
    constraint_bounds = TwiceDifferentiableConstraints(
        lower_bounds,
        upper_bounds,
        x -> [budget_constraint(x)],  # Inequality constraints g(x) ≥ 0
        [0.0],  # Lower bound for constraints
        [Inf]   # Upper bound for constraints
    )

    result = optimize(
        constraints,
        constraint_bounds,
        initial_guess,
        IPNewton()
    )
    
    C_t, L_t = Optim.minimizer(result)
    return C_t, L_t
end

# Modified firm decision to use optimization
function firm_decision(w_t, r_t, τ, A_eta_samples)
    n = size(A_eta_samples, 1)
    results = Vector{NamedTuple{(:A_t, :η_t, :K_t, :L_t, :Y_t, :profit), 
                               Tuple{Float64,Float64,Float64,Float64,Float64,Float64}}}(undef, n)
    
    for i in 1:n
        A_t, η_t = A_eta_samples[i, :]
        K_t, L_t, Y_t, profit = firm_optimize(w_t, r_t, τ, A_t, η_t)
        results[i] = (A_t=A_t, η_t=η_t, K_t=K_t, L_t=L_t, Y_t=Y_t, profit=profit)
    end
    
    # Choose the (A_t, η_t) that maximizes profits
    max_idx = argmax(r -> r.profit, results)
    return results[max_idx]
end

# Modified equilibrium computation
function compute_equilibrium(τ)
    function equilibrium_conditions(prices)
        w_t, r_t = prices
        
        # Sample technology and emissions intensity pairs
        A_eta_samples = sample_A_eta(100)  # Reduced sample size for computational efficiency
        
        # Firms' optimal decisions
        firm_result = firm_decision(w_t, r_t, τ, A_eta_samples)
        
        # Households' optimal decisions
        initial_assets = firm_result.K_t  # Assume households own the capital stock
        C_t, L_supply = household_optimize(w_t, r_t, initial_assets)
        
        # Market clearing conditions
        labor_market_error = L_supply - firm_result.L_t
        capital_market_error = initial_assets - firm_result.K_t
        
        return [labor_market_error, capital_market_error]
    end

    # Solve for equilibrium prices
    prices_guess = [1.0, 0.05]
    sol = nlsolve(equilibrium_conditions, prices_guess)
    
    if !converged(sol)
        error("Equilibrium computation did not converge")
    end
    
    # Compute final equilibrium values
    w_t, r_t = sol.zero
    A_eta_samples = sample_A_eta(100)
    result = firm_decision(w_t, r_t, τ, A_eta_samples)
    C_t, L_t = household_optimize(w_t, r_t, result.K_t)
    
    return Dict(
        "w_t" => w_t,
        "r_t" => r_t,
        "A_t" => result.A_t,
        "η_t" => result.η_t,
        "K_t" => result.K_t,
        "L_t" => result.L_t,
        "Y_t" => result.Y_t,
        "C_t" => C_t
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
