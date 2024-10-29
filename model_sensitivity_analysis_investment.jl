using JuMP
using Ipopt
using Statistics
using Distributions
using DataFrames
using CSV
using CairoMakie
include("model_core_partial_investment.jl")
using .ModelParametersModule: ModelParameters

function run_sensitivity_analysis()
    # Base parameters - create new instance for each scale
    function create_scaled_params(scale)
        return ModelParameters(
            β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
            δ = 0.1, γ = 0.01, K_init = 1.0,
            μ_A = 1.0 * scale,  # Scale mean productivity
            μ_η = 1.0, 
            σ_A = 0.2 * scale,  # Scale variance proportionally
            σ_η = 0.2, 
            ρ = 0.5,
            θ_min = 0.1, 
            θ_max = 1.0
        )
    end

    # Enhanced results DataFrame to capture technology split
    results = DataFrame(
        A_0 = Float64[], η_0 = Float64[], 
        A_1 = Float64[], η_1 = Float64[],
        skill_factor = Float64[],
        success = Bool[],
        labor_efficiency = Float64[],
        Y_0 = Float64[], Y_1 = Float64[],
        tech_adoption_rate = Float64[],
        A_1_eff = Float64[],
        η_1_eff = Float64[],
        error_type = String[]
    )

    # Parameter ranges to test
    skill_factors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    distribution_scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]  # Scales for the joint distribution
    n_samples = 10  # Number of samples per configuration

    println("Starting sensitivity analysis...")
    total_tests = length(skill_factors) * length(distribution_scales) * n_samples
    current_test = 0

    for scale in distribution_scales
        # Create new parameters with scaled values
        modified_params = create_scaled_params(scale)

        for sf in skill_factors
            for sample in 1:n_samples
                current_test += 1
                println("Running test $current_test/$total_tests")
                println("Distribution scale: $scale, skill_factor: $sf, sample: $sample")

                # Declare variables in outer scope
                local A_0, η_0, A_1, η_1
                
                try
                    # Use the existing sampling function
                    A_0, η_0, A_1, η_1 = sample_technology(modified_params)
                    
                    result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, sf, modified_params)
                    
                    push!(results, (
                        A_0, η_0, A_1, η_1, sf, true,
                        result["Labor_Efficiency"],
                        result["Y_0"],
                        result["Y_1"],
                        result["Technology_Split"],
                        result["A_1_eff"],
                        result["η_1_eff"],
                        "none"
                    ))
                catch e
                    # Handle the case where sampling failed
                    if !(@isdefined A_0)
                        A_0, η_0, A_1, η_1 = NaN, NaN, NaN, NaN
                    end
                    
                    error_type = if isa(e, ErrorException)
                        if contains(string(e), "LOCALLY_INFEASIBLE")
                            "infeasible"
                        elseif contains(string(e), "ITERATION_LIMIT")
                            "iteration_limit"
                        else
                            string(typeof(e))
                        end
                    else
                        string(typeof(e))
                    end

                    push!(results, (
                        A_0, η_0, A_1, η_1, sf, false,
                        NaN, NaN, NaN, NaN, NaN, NaN,
                        error_type
                    ))
                end
            end
        end
    end

    # Analysis of results
    println("\nAnalysis Results:")
    
    # Success rate by distribution scale
    println("\nSuccess rate by distribution scale:")
    for scale in distribution_scales
        # Use a tolerance band around the scale value
        scale_lower = scale - 0.05
        scale_upper = scale + 0.05
        
        subset = results[(results.A_1 ./ results.A_0 .>= scale_lower) .& 
                        (results.A_1 ./ results.A_0 .< scale_upper), :]
        
        if !isempty(subset)
            success_rate = mean(subset.success) * 100
            successful_cases = subset[subset.success, :]
            
            if !isempty(successful_cases)
                avg_adoption = mean(successful_cases.tech_adoption_rate) * 100
                println("Scale = $scale: $(round(success_rate, digits=1))% success, " *
                       "$(round(avg_adoption, digits=1))% average adoption")
                
                # Additional statistics
                println("  Average Labor Efficiency: $(round(mean(successful_cases.labor_efficiency), digits=3))")
                println("  Average Y1/Y0 ratio: $(round(mean(successful_cases.Y_1 ./ successful_cases.Y_0), digits=3))")
            else
                println("Scale = $scale: $(round(success_rate, digits=1))% success, no successful cases")
            end
        else
            println("Scale = $scale: No cases found in this scale range")
        end
    end

    # Generate visualizations
    println("\nGenerating visualizations...")
    
    fig = Figure(size=(1200, 800))
    
    # 1. Success Rate by Distribution Scale and Skill Factor
    ax1 = Axis(fig[1, 1], 
        title = "Success Rate",
        xlabel = "Distribution Scale",
        ylabel = "Skill Factor"
    )

    # 2. Technology Adoption Rate
    ax2 = Axis(fig[1, 2],
        title = "Technology Adoption Rate",
        xlabel = "Distribution Scale",
        ylabel = "Skill Factor"
    )

    # Prepare data for heatmaps
    success_matrix = zeros(length(skill_factors), length(distribution_scales))
    adoption_matrix = zeros(length(skill_factors), length(distribution_scales))
    
    for (i, sf) in enumerate(skill_factors)
        for (j, scale) in enumerate(distribution_scales)
            subset = results[(results.skill_factor .== sf) .&& 
                           (results.A_1 ./ results.A_0 .≈ scale), :]
            success_matrix[i, j] = mean(subset.success)
            adoption_matrix[i, j] = mean(subset[subset.success, :tech_adoption_rate])
        end
    end

    # Create heatmaps
    hm1 = heatmap!(ax1, success_matrix,
        colormap = :viridis,
        interpolate = false
    )
    
    hm2 = heatmap!(ax2, adoption_matrix,
        colormap = :viridis,
        interpolate = false
    )

    # Add colorbars
    Colorbar(fig[1, 3], hm1, label = "Success Rate")
    Colorbar(fig[1, 4], hm2, label = "Adoption Rate")

    # Add tick labels
    for ax in [ax1, ax2]
        ax.xticks = (1:length(distribution_scales), string.(distribution_scales))
        ax.yticks = (1:length(skill_factors), string.(skill_factors))
    end

    # Save results
    println("\nSaving results...")
    CSV.write("sensitivity_results_investment.csv", results)
    save("sensitivity_analysis_investment.png", fig, px_per_unit=2)

    return results, fig
end

# Run the analysis
results, fig = run_sensitivity_analysis()