using JuMP, Ipopt, Statistics, Distributions, DataFrames, CSV, CairoMakie
include("model_core_partial_investment.jl")
using .ModelParametersModule: ModelParameters

function run_sensitivity_analysis()
    # Create results DataFrame with focus on failure patterns
    results = DataFrame(
        τ = Float64[],           # Tax rate
        η = Float64[],           # Carbon intensity
        τ_x_η = Float64[],       # Tax * Carbon intensity interaction
        skill_factor = Float64[],
        A = Float64[],           # Productivity
        success = Bool[],
        effective_output = Float64[], # (1 - τ*η) * Y_0
        output_multiplier = Float64[], # (1 - τ*η)
        error_type = String[]
    )

    # Test ranges focused on observed failure zones
    tax_rates = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    η_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]  # Include problematic η ≈ 1.64
    skill_factors = [0.5, 1.0, 1.5, 2.0, 2.56, 3.0]  # Include problematic sf ≈ 2.56
    A_values = [0.8, 1.0, 1.2, 1.48, 1.6]  # Include problematic A ≈ 1.48

    # Base parameters
    base_params = ModelParameters()
    
    println("Starting sensitivity analysis...")
    total_tests = length(tax_rates) * length(η_values) * length(skill_factors)
    current_test = 0

    # Test 1: Tax and Carbon Intensity Interaction
    println("\nTesting Tax and Carbon Intensity Interaction...")
    for τ in tax_rates, η in η_values
        current_test += 1
        println("Running test $current_test/$total_tests: τ=$τ, η=$η")
        
        try
            result = compute_equilibrium_core(
                τ, 1.0, η, 1.0, η, 1.0, base_params
            )
            
            push!(results, (
                τ, η, τ*η, 1.0, 1.0, true,
                result["Y_0"] * (1 - τ*η),
                (1 - τ*η),
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
                τ, η, τ*η, 1.0, 1.0, false,
                NaN, (1 - τ*η),
                error_type
            ))
        end
    end

    # Test 2: High Carbon Intensity with Varying Skill Factor
    println("\nTesting High Carbon Intensity with Varying Skill Factor...")
    for sf in skill_factors, η in η_values
        try
            result = compute_equilibrium_core(
                0.15, 1.0, η, 1.0, η, sf, base_params
            )
            
            push!(results, (
                0.15, η, 0.15*η, sf, 1.0, true,
                result["Y_0"] * (1 - 0.15*η),
                (1 - 0.15*η),
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
                0.15, η, 0.15*η, sf, 1.0, false,
                NaN, (1 - 0.15*η),
                error_type
            ))
        end
    end

    # Analysis
    println("\nAnalyzing results...")
    
    # Find critical thresholds
    successful = results[results.success .== true, :]
    failed = results[results.success .== false, :]
    
    if !isempty(successful) && !isempty(failed)
        critical_τη = mean([maximum(successful.τ_x_η), minimum(failed.τ_x_η)])
        println("\nCritical τ*η threshold: ", round(critical_τη, digits=3))
        
        # Analyze skill factor impact
        high_sf = results[results.skill_factor .> 2.0, :]
        low_sf = results[results.skill_factor .<= 2.0, :]
        println("Success rate with high skill factor (>2.0): ", 
                round(mean(high_sf.success) * 100, digits=1), "%")
        println("Success rate with low skill factor (≤2.0): ", 
                round(mean(low_sf.success) * 100, digits=1), "%")
    end

    # Visualizations
    println("\nGenerating visualizations...")
    
    fig = Figure(size=(1500, 1000))

    # 1. Success rate heatmap for τ vs η
    ax1 = Axis(fig[1, 1], 
        title = "Success Rate: Tax vs Carbon Intensity",
        xlabel = "Tax Rate",
        ylabel = "Carbon Intensity"
    )

    # Create success rate matrix
    success_matrix = zeros(length(tax_rates), length(η_values))
    for (i, τ) in enumerate(tax_rates), (j, η) in enumerate(η_values)
        subset = results[(results.τ .== τ) .&& (results.η .== η), :]
        success_matrix[i, j] = mean(subset.success)
    end

    hm1 = heatmap!(ax1, success_matrix,
        colormap = :viridis,
        interpolate = true
    )
    Colorbar(fig[1, 2], hm1, label = "Success Rate")

    # 2. Success rate heatmap for skill factor vs η
    ax2 = Axis(fig[1, 3],
        title = "Success Rate: Skill Factor vs Carbon Intensity",
        xlabel = "Skill Factor",
        ylabel = "Carbon Intensity"
    )

    # Create second success rate matrix
    success_matrix2 = zeros(length(skill_factors), length(η_values))
    for (i, sf) in enumerate(skill_factors), (j, η) in enumerate(η_values)
        subset = results[(results.skill_factor .== sf) .&& (results.η .== η), :]
        success_matrix2[i, j] = mean(subset.success)
    end

    hm2 = heatmap!(ax2, success_matrix2,
        colormap = :viridis,
        interpolate = true
    )
    Colorbar(fig[1, 4], hm2, label = "Success Rate")

    # 3. Effective output vs τ*η scatter
    ax3 = Axis(fig[2, 1:2],
        title = "Effective Output vs τ*η",
        xlabel = "τ*η",
        ylabel = "Effective Output"
    )

    successful_points = scatter!(ax3, 
        successful.τ_x_η, successful.effective_output,
        color = :blue, label = "Successful"
    )

    # Add vertical line at critical threshold
    if @isdefined critical_τη
        vlines!(ax3, [critical_τη], color = :red, linestyle = :dash,
                label = "Critical Threshold")
    end

    # 4. Output multiplier distribution
    ax4 = Axis(fig[2, 3:4],
        title = "Distribution of Output Multiplier (1 - τ*η)",
        xlabel = "Output Multiplier",
        ylabel = "Count"
    )

    hist!(ax4, successful.output_multiplier, bins=30, 
          color=(:blue, 0.5), label="Successful Cases")

    # Save results
    println("\nSaving results...")
    CSV.write("sensitivity_results_investment.csv", results)
    save("sensitivity_analysis_investment.png", fig, px_per_unit=2)

    return results, fig
end

# Run the analysis
results, fig = run_sensitivity_analysis()