using JuMP, Ipopt, Statistics, DataFrames, CSV, CairoMakie
using StatsBase  # For weighted sampling
using Distributions  # Import the package
using StatsBase: quantile, cut

include("model_core_partial_investment.jl")
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS

function bounded_normal(μ, σ, lower, upper)
    # Create a normal distribution and use rejection sampling
    # to generate samples within bounds
    d = Normal(μ, σ)
    x = rand(d)
    while !(lower ≤ x ≤ upper)
        x = rand(d)
    end
    return x
end

function generate_parameter_sample(n_samples::Int)
    # Define parameter ranges and distributions
    param_distributions = Dict(
        # Tax and carbon intensity parameters
        :τ => Uniform(0.0, 0.3),
        :η => () -> bounded_normal(1.0, 0.4, 0.5, 2.0),
        :skill_factor => () -> bounded_normal(1.5, 0.75, 0.1, 4.0),
        
        # Technology parameters
        :A => () -> bounded_normal(1.0, 0.3, 0.5, 2.0),
        
        # Model parameters (variations around defaults)
        :β => () -> bounded_normal(0.96, 0.02, 0.9, 0.99),
        :σ => () -> bounded_normal(2.0, 0.5, 1.0, 4.0),
        :α => () -> bounded_normal(0.33, 0.05, 0.2, 0.5),
        :γ => () -> bounded_normal(0.01, 0.005, 0.001, 0.05)
    )
    
    # Generate samples
    samples = DataFrame(
        τ = rand(param_distributions[:τ], n_samples),
        η = [param_distributions[:η]() for _ in 1:n_samples],
        skill_factor = [param_distributions[:skill_factor]() for _ in 1:n_samples],
        A = [param_distributions[:A]() for _ in 1:n_samples],
        β = [param_distributions[:β]() for _ in 1:n_samples],
        σ = [param_distributions[:σ]() for _ in 1:n_samples],
        α = [param_distributions[:α]() for _ in 1:n_samples],
        γ = [param_distributions[:γ]() for _ in 1:n_samples]
    )
    
    # Add derived parameters
    samples.τ_x_η = samples.τ .* samples.η
    
    return samples
end


function run_monte_carlo_analysis(n_samples::Int = 1000)
    # Generate parameter samples
    println("Generating $n_samples parameter combinations...")
    samples = generate_parameter_sample(n_samples)
    
    # Create results DataFrame
    results = DataFrame(
        τ = Float64[],
        η = Float64[],
        τ_x_η = Float64[],
        skill_factor = Float64[],
        A = Float64[],
        β = Float64[],
        σ = Float64[],
        α = Float64[],
        γ = Float64[],
        success = Bool[],
        effective_output = Float64[],
        output_multiplier = Float64[],
        labor_efficiency = Float64[],
        technology_split = Float64[],
        error_type = String[]
    )
    
    println("Running Monte Carlo simulation...")
    for i in 1:n_samples
        if i % 100 == 0
            println("Processing sample $i/$n_samples")
        end
        
        # Create custom parameters for this run
        custom_params = ModelParameters(
            β = samples[i, :β],
            σ = samples[i, :σ],
            α = samples[i, :α],
            γ = samples[i, :γ]
        )
        
        try
            result = compute_equilibrium_core(
                samples[i, :τ],
                samples[i, :A],
                samples[i, :η],
                samples[i, :A],
                samples[i, :η],
                samples[i, :skill_factor],
                custom_params
            )
            
            push!(results, (
                samples[i, :τ],
                samples[i, :η],
                samples[i, :τ_x_η],
                samples[i, :skill_factor],
                samples[i, :A],
                samples[i, :β],
                samples[i, :σ],
                samples[i, :α],
                samples[i, :γ],
                true,
                result["Y_0"] * (1 - samples[i, :τ] * samples[i, :η]),
                (1 - samples[i, :τ] * samples[i, :η]),
                round(result["Labor_Efficiency"], digits=5),
                result["Technology_Split"],
                "none"
            ))
        catch e
            error_type = if contains(string(e), "LOCALLY_INFEASIBLE")
                "infeasible"
            elseif contains(string(e), "ALMOST_LOCALLY_SOLVED")
                "almost_solved"
            else
                string(typeof(e))
            end
            
            push!(results, (
                samples[i, :τ],
                samples[i, :η],
                samples[i, :τ_x_η],
                samples[i, :skill_factor],
                samples[i, :A],
                samples[i, :β],
                samples[i, :σ],
                samples[i, :α],
                samples[i, :γ],
                false,
                NaN,
                (1 - samples[i, :τ] * samples[i, :η]),
                NaN,
                NaN,
                error_type
            ))
        end
    end

    # After the simulation loop, save results to CSV
    output_file = "monte_carlo_results.csv"
    CSV.write(output_file, results)
    println("\nResults saved to $output_file")
    
    return results
end

# Run the analysis if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_monte_carlo_analysis(5000)
end

