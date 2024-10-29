using JuMP
using Ipopt
using Statistics
using Distributions
using DataFrames
using CSV
using CairoMakie
include("model_core_skill_fix.jl")

function run_sensitivity_analysis()
    # Base parameters
    base_params = ModelParameters(
        β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
        δ = 0.1, γ = 0.01, K_init = 1.0,
        μ_A = 1.0, μ_η = 1.0, σ_A = 0.2, σ_η = 0.2, ρ = 0.5,
        θ_min = 0.1, θ_max = 1.0
    )

    results = DataFrame(
        A_0 = Float64[], η_0 = Float64[], 
        A_1 = Float64[], η_1 = Float64[],
        skill_factor = Float64[],
        success = Bool[],
        labor_efficiency = Float64[],
        Y_0 = Float64[],
        L_0 = Float64[],
        error_type = String[]
    )

    # Parameter ranges to test
    relative_A_range = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]  # These become multipliers
    η_changes = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
    skill_factors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    println("Starting sensitivity analysis...")
    total_tests = length(relative_A_range) * length(η_changes) * length(skill_factors)
    current_test = 0

    for relative_A in relative_A_range
        A_0 = 1.0  # Base productivity always 1.0
        A_1 = relative_A  # Testing relative changes

        for Δη in η_changes
            η_0 = 1.0  # Base carbon intensity
            η_1 = η_0 + Δη

            for sf in skill_factors
                current_test += 1
                println("Running test $current_test/$total_tests")
                println("A_0: $A_0, Δη: $Δη, skill_factor: $sf")

                try
                    result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, sf, base_params)
                    
                    push!(results, (
                        A_0, η_0, A_1, η_1, sf, true,
                        result["Labor_Efficiency"],
                        result["Y_0"],
                        result["L_0"],
                        "none"
                    ))
                catch e
                    error_type = if isa(e, ErrorException) && contains(string(e), "LOCALLY_INFEASIBLE")
                        "infeasible"
                    else
                        string(typeof(e))
                    end

                    push!(results, (
                        A_0, η_0, A_1, η_1, sf, false,
                        NaN, NaN, NaN,
                        error_type
                    ))
                end
            end
        end
    end

    # Analysis of results
    println("\nAnalysis Results:")
    
    # Success rate by productivity level
    println("\nSuccess rate by productivity level:")
    for A_0 in relative_A_range
        success_rate = mean(results[results.A_0 .== A_0, :success]) * 100
        println("A_0 = $A_0: $(round(success_rate, digits=1))%")
    end

    # Success rate by technology change
    println("\nSuccess rate by technology change (Δη):")
    for Δη in η_changes
        subset = results[results.η_1 .== (1.0 + Δη), :]
        success_rate = mean(subset.success) * 100
        println("Δη = $Δη: $(round(success_rate, digits=1))%")
    end

    # Success rate by skill factor
    println("\nSuccess rate by skill factor:")
    for sf in skill_factors
        success_rate = mean(results[results.skill_factor .== sf, :success]) * 100
        println("skill_factor = $sf: $(round(success_rate, digits=1))%")
    end

    # Generate heatmaps
    println("\nGenerating visualization...")
    
    # Create figure with explicit assignment using CairoMakie
    fig = Figure(size=(800, 600))  # Changed from resolution to size
    
    # Create main axis
    ax = Axis(fig[1, 1], 
        title = "Success Rate by A₀ and Δη",
        xlabel = "Technology Change (Δη)",
        ylabel = "Productivity (A₀)"
    )

    # Prepare data for heatmap
    success_matrix = zeros(length(relative_A_range), length(η_changes))
    for (i, A_0) in enumerate(relative_A_range)
        for (j, Δη) in enumerate(η_changes)
            subset = results[(results.A_0 .== A_0), :]
            success_matrix[i, j] = mean(subset.success)
        end
    end

    # Create heatmap using CairoMakie without interpolation
    hm = CairoMakie.heatmap!(ax, 
        1:length(η_changes),  # Use indices instead of actual values
        1:length(relative_A_range), 
        success_matrix,
        colormap = :viridis,
        interpolate = false  # Disable interpolation
    )

    # Add colorbar
    Colorbar(fig[1, 2], hm, 
        label = "Success Rate",
        width = 15
    )

    # Add tick labels
    ax.xticks = (1:length(η_changes), string.(round.(η_changes, digits=2)))
    ax.yticks = (1:length(relative_A_range), string.(round.(relative_A_range, digits=2)))

    # Adjust layout
    fig[1, 1] = ax
    colgap!(fig.layout, 10)

    # Save results
    println("\nSaving results...")
    CSV.write("sensitivity_results.csv", results)
    
    # Save the figure
    save("sensitivity_heatmap.png", fig, px_per_unit=2)  # Added px_per_unit for better resolution

    # Identify boundary cases
    println("\nBoundary Cases (where success/failure transitions):")
    for A_0 in relative_A_range
        for Δη in η_changes
            subset = results[(results.A_0 .== A_0), :]
            if 0.0 < mean(subset.success) < 1.0
                println("Boundary at A_0 = $A_0, Δη = $Δη")
                println("Success rate: $(mean(subset.success) * 100)%")
            end
        end
    end

    return results, fig
end

# Run the analysis
results, fig = run_sensitivity_analysis()