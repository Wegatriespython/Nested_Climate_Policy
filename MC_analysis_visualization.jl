# Add at the beginning of the file:
using Statistics, DataFrames, CSV, CairoMakie
using StatsBase  # For weighted sampling
using Distributions  # Import the package


function load_monte_carlo_results(filepath::String="monte_carlo_results.csv")
    results = CSV.read(filepath, DataFrame)
    
    # Convert string success column to boolean if needed
    if eltype(results.success) == String
        results.success = parse.(Bool, results.success)
    end
    
    return results
end

# Add this new function after analyze_advanced_interactions but before run_visualization_analysis
function analyze_failure_modes(results::DataFrame)
    # Separate successful and failed runs
    successful = results[results.success .== true, :]
    failed = results[results.success .== false, :]
    
    println("\nFailure Mode Analysis:")
    println("----------------------")
    println("Total runs: $(nrow(results))")
    println("Success rate: $(round(100 * nrow(successful) / nrow(results), digits=1))%")
    
    # Find critical thresholds
    critical_points = Dict{Symbol, NamedTuple}()
    
    for param in [:τ, :η, :τ_x_η, :skill_factor, :A]
        if !isempty(successful) && !isempty(failed)
            success_95th = quantile(successful[:, param], 0.95)
            success_5th = quantile(successful[:, param], 0.05)
            fail_median = median(failed[:, param])
            
            critical_points[param] = (
                success_range = (success_5th, success_95th),
                fail_median = fail_median
            )
            
            println("\nParameter: $param")
            println("  Success range (5th-95th percentile): $(round(success_5th, digits=3)) - $(round(success_95th, digits=3))")
            println("  Failure median: $(round(fail_median, digits=3))")
        end
    end
    
    # Identify common failure combinations
    println("\nCommon failure combinations:")
    
    # Group failures by ranges of τ_x_η
    τη_ranges = range(0, maximum(results.τ_x_η), length=6)
    for i in 1:length(τη_ranges)-1
        range_failed = failed[τη_ranges[i] .<= failed.τ_x_η .< τη_ranges[i+1], :]
        if nrow(range_failed) > 0
            println("\nτ*η range: $(round(τη_ranges[i], digits=2)) - $(round(τη_ranges[i+1], digits=2))")
            println("  Number of failures: $(nrow(range_failed))")
            println("  Average skill factor: $(round(mean(range_failed.skill_factor), digits=3))")
            println("  Average A: $(round(mean(range_failed.A), digits=3))")
        end
    end
    
    # Identify parameter interactions
    println("\nParameter interactions in failures:")
    interaction_params = [:τ_x_η, :skill_factor, :A]
    
    for i in 1:length(interaction_params)
        for j in i+1:length(interaction_params)
            p1, p2 = interaction_params[i], interaction_params[j]
            correlation = cor(failed[:, p1], failed[:, p2])
            println("$p1 vs $p2 correlation in failures: $(round(correlation, digits=3))")
        end
    end
    
    return critical_points
end

function analyze_advanced_interactions(results::DataFrame)
    successful = results[results.success .== true, :]
    failed = results[results.success .== false, :]
    
    println("\nAdvanced Parameter Interaction Analysis:")
    println("--------------------------------------")
    
    # 1. Three-way parameter interactions
    println("\n1. Three-way Parameter Interactions:")
    three_way_params = [(:τ_x_η, :skill_factor, :A), 
                       (:τ_x_η, :σ, :α),
                       (:skill_factor, :γ, :σ)]
    
    for (p1, p2, p3) in three_way_params
        # Calculate success rate in different octants of parameter space
        median_p1 = median(results[:, p1])
        median_p2 = median(results[:, p2])
        median_p3 = median(results[:, p3])
        
        # Define regions
        region_mask = Dict{String, Vector{Bool}}()
        region_mask["low_all"] = (results[:, p1] .≤ median_p1) .& 
                                (results[:, p2] .≤ median_p2) .& 
                                (results[:, p3] .≤ median_p3)
        region_mask["high_all"] = (results[:, p1] .> median_p1) .& 
                                 (results[:, p2] .> median_p2) .& 
                                 (results[:, p3] .> median_p3)
        
        println("\nInteraction between $p1, $p2, and $p3:")
        for (region_name, mask) in region_mask
            success_rate = mean(results[mask, :success]) * 100
            println("  Success rate in $region_name: $(round(success_rate, digits=1))%")
        end
    end
    
    # 2. Output sensitivity analysis
    println("\n2. Output Sensitivity Analysis:")
    successful_only = results[results.success .== true, :]
    
    if !isempty(successful_only)
        output_correlations = Dict()
        for param in [:τ_x_η, :skill_factor, :A, :σ, :α, :γ]
            correlation = cor(successful_only[:, param], successful_only.effective_output)
            output_correlations[param] = correlation
            println("Correlation of $param with effective output: $(round(correlation, digits=3))")
        end
        
        # Find strongest output determinants
        sorted_impacts = sort(collect(output_correlations), by=x->abs(x[2]), rev=true)
        println("\nParameters ranked by impact on output:")
        for (param, impact) in sorted_impacts
            println("  $param: $(round(impact, digits=3))")
        end
    end
    
    # 3. Technology adoption analysis
    println("\n3. Technology Adoption Analysis:")
    if !isempty(successful_only)
        tech_correlations = Dict()
        for param in [:τ_x_η, :skill_factor, :A, :σ, :α, :γ]
            correlation = cor(successful_only[:, param], successful_only.technology_split)
            tech_correlations[param] = correlation
            println("Correlation of $param with technology split: $(round(correlation, digits=3))")
        end
    end
    
    # 4. Labor efficiency determinants
    println("\n4. Labor Efficiency Analysis:")
    if !isempty(successful_only)
        # Calculate elasticities
        for param in [:τ_x_η, :skill_factor, :A]
            # Use log differences for approximate elasticity
            param_pct_change = (successful_only[:, param] .- mean(successful_only[:, param])) ./ mean(successful_only[:, param])
            efficiency_pct_change = (successful_only.labor_efficiency .- mean(successful_only.labor_efficiency)) ./ mean(successful_only.labor_efficiency)
            
            # Calculate average elasticity
            elasticity = mean(efficiency_pct_change ./ param_pct_change)
            println("Labor efficiency elasticity to $param: $(round(elasticity, digits=3))")
        end
    end
    
    # 5. Failure mode clustering
    println("\n5. Failure Mode Clustering:")
    if nrow(failed) > 10  # Only if we have enough failures
        # Simple clustering based on τ_x_η ranges
        τη_clusters = quantile(failed.τ_x_η, [0.33, 0.67])
        
        println("Low τ*η failures (≤ $(round(τη_clusters[1], digits=3))):")
        low_failures = failed[failed.τ_x_η .<= τη_clusters[1], :]
        println("  Average skill factor: $(round(mean(low_failures.skill_factor), digits=3))")
        println("  Average A: $(round(mean(low_failures.A), digits=3))")
        
        println("High τ*η failures (> $(round(τη_clusters[2], digits=3))):")
        high_failures = failed[failed.τ_x_η .> τη_clusters[2], :]
        println("  Average skill factor: $(round(mean(high_failures.skill_factor), digits=3))")
        println("  Average A: $(round(mean(high_failures.A), digits=3))")
    end
    
    return Dict(
        "output_correlations" => output_correlations,
        "tech_correlations" => tech_correlations
    )
end

function generate_productivity_controlled_heatmaps(results::DataFrame)
    println("\nGenerating productivity-controlled heatmaps...")
    
    fig_productivity = Figure(size=(1800, 600), fontsize=12)

    # Split productivity into three levels (low, medium, high)
    A_quantiles = quantile(results.A, [0.33, 0.67])
    A_levels = ["Low A (≤$(round(A_quantiles[1], digits=2)))",
                "Med A ($(round(A_quantiles[1], digits=2))-$(round(A_quantiles[2], digits=2)))",
                "High A (≥$(round(A_quantiles[2], digits=2)))"]

    # Setup ranges for heatmaps
    skill_range = range(minimum(results.skill_factor), maximum(results.skill_factor), length=50)
    carbon_range = range(minimum(results.η), maximum(results.η), length=50)

    success_matrices = []
    for A_range in [
        results.A .<= A_quantiles[1],
        A_quantiles[1] .< results.A .<= A_quantiles[2],
        results.A .> A_quantiles[2]
    ]
        success_matrix = zeros(50, 50)
        filtered_results = results[A_range, :]
        
        for (i, skill) in enumerate(skill_range), (j, carbon) in enumerate(carbon_range)
            nearby = findall(
                (abs.(filtered_results.skill_factor .- skill) .< 0.1) .&
                (abs.(filtered_results.η .- carbon) .< 0.1)
            )
            success_matrix[i, j] = isempty(nearby) ? 0.0 : mean(filtered_results.success[nearby])
        end
        push!(success_matrices, success_matrix)
    end

    # Plot the controlled heatmaps side by side
    for (i, (matrix, title)) in enumerate(zip(success_matrices, A_levels))
        ax = Axis(fig_productivity[1, i],
            title = title,
            xlabel = "Skill Factor",
            ylabel = "Carbon Intensity"
        )
        
        hm = heatmap!(ax, skill_range, carbon_range, matrix, colormap=:viridis)
        Colorbar(fig_productivity[1, i+3], hm, label="Success Rate")
    end

    save("monte_carlo_productivity_controlled.png", fig_productivity, px_per_unit=2)
    
    return fig_productivity
end

# Modify the main analysis function:
function run_visualization_analysis()
    println("Loading Monte Carlo results...")
    results = load_monte_carlo_results()
    
    # Analysis
    println("\nAnalyzing results...")
    
    # Find critical thresholds using quantile analysis
    successful = results[results.success .== true, :]
    failed = results[results.success .== false, :]
    
    if !isempty(successful) && !isempty(failed)
        critical_τη = quantile(successful.τ_x_η, 0.95)
        println("\nCritical τ*η threshold (95th percentile of successful runs): ", 
                round(critical_τη, digits=3))
        
        # Parameter importance analysis
        for param in [:τ, :η, :skill_factor, :A, :β, :σ, :α, :γ]
            success_rate_high = mean(results[results[:, param] .> median(results[:, param]), :success])
            success_rate_low = mean(results[results[:, param] .<= median(results[:, param]), :success])
            
            println("Parameter $param impact:")
            println("  Success rate above median: $(round(success_rate_high * 100, digits=1))%")
            println("  Success rate below median: $(round(success_rate_low * 100, digits=1))%")
        end
    end

    # Visualizations
    println("\nGenerating visualizations...")
    
    # Create main figure with adjusted size and spacing
    fig = Figure(size=(1800, 1200), fontsize=12)

    # 1. Parameter correlation heatmap (existing)
    ax1 = Axis(fig[1, 1], 
        title = "Parameter Correlations with Success",
        xlabel = "Parameters",
        ylabel = "Parameters"
    )

    params = [:τ, :η, :skill_factor, :A, :β, :σ, :α, :γ]
    cor_matrix = zeros(length(params), length(params))
    for (i, p1) in enumerate(params), (j, p2) in enumerate(params)
        cor_matrix[i, j] = cor(results[:, p1], results[:, p2])
    end

    hm1 = heatmap!(ax1, cor_matrix,
        colormap = :seismic,
        colorrange = (-1, 1)
    )
    Colorbar(fig[1, 2], hm1, label = "Correlation")

    # 2. Success probability scatter (existing, but improved)
    ax2 = Axis(fig[1, 3:4],
        title = "Success Probability",
        xlabel = "τ*η",
        ylabel = "Skill Factor"
    )

    scatter!(ax2,
        successful.τ_x_η, successful.skill_factor,
        color = :blue, alpha = 0.3, label = "Success",
        markersize = 8
    )
    scatter!(ax2,
        failed.τ_x_η, failed.skill_factor,
        color = :red, alpha = 0.3, label = "Failure",
        markersize = 8
    )
    axislegend(ax2, position = :lt)


       # 3. Parameter distributions (improved spacing)
    for (i, param) in enumerate([:τ, :η, :A])
        ax = Axis(fig[2, i],
            title = "Distribution of $param",
            xlabel = "$param",
            ylabel = "Density",
            width = 500,  # Set explicit width
            height = 500  # Set explicit height
        )
        
        if param == :η
            limits!(ax, 0, 2.0, 0, 1.5)  # Set explicit limits for η
        end
        
        density!(ax, successful[:, param], 
            color = (:blue, 0.3), 
            label = "Success",
            bandwidth = 0.02
        )
        density!(ax, failed[:, param], 
            color = (:red, 0.3), 
            label = "Failure",
            bandwidth = 0.02
        )
            axislegend(ax, position = :lt)
    end
    
    # Create additional figure for new visualizations
    fig_extra = Figure(size=(1800, 1200), fontsize=12)

    # 4. Adjustment Cost vs Output
    ax_eff = Axis(fig_extra[1, 1],
        title = "Adjustment Cost vs Output",
        xlabel = "Output (Y₀)",
        ylabel = "Adjustment Cost (γ)"
    )
    
    limits!(ax_eff, nothing, nothing, 0.00001, 0.05)  # Set y-axis limits
    
    scatter!(ax_eff, 
        successful.effective_output, successful.γ,
        color = successful.labor_efficiency,
        colormap = :viridis,
        markersize = 8
    )
    Colorbar(fig_extra[1, 2], colormap = :viridis, label = "Labor Efficiency")

    # 5. Effective Output vs τ*η
    ax_out = Axis(fig_extra[1, 3],
        title = "Effective Output vs τ*η",
        xlabel = "τ*η",
        ylabel = "Effective Output"
    )
    
    scatter!(ax_out, successful.τ_x_η, successful.effective_output,
        color = :blue, alpha = 0.5,
        markersize = 8
    )

    # 6. Distribution of Output Multiplier
    ax_mult = Axis(fig_extra[2, 1:2],
        title = "Distribution of Output Multiplier (1 - τ*η)",
        xlabel = "Output Multiplier",
        ylabel = "Count"
    )
    
    hist!(ax_mult, successful.output_multiplier, 
        bins = 30,
        color = (:blue, 0.5)
    )

    # Create additional figure for distribution plots
    fig_dist = Figure(size=(1800, 1200), fontsize=12)

    # Add distribution plots matching your reference images
    # 1. Distribution of τ
    ax_dist1 = Axis(fig_dist[2, 1],
        title = "Distribution of τ",
        xlabel = "τ",
        ylabel = "Density"
    )
    density!(ax_dist1, successful.τ, label="Success", color=(:blue, 0.5))
    density!(ax_dist1, failed.τ, label="Failure", color=(:red, 0.5))
    axislegend(ax_dist1)

    # 2. Distribution of η
    ax_dist2 = Axis(fig_dist[2, 2],
        title = "Distribution of η",
        xlabel = "η",
        ylabel = "Density"
    )
    density!(ax_dist2, successful.η, label="Success", color=(:blue, 0.5))
    density!(ax_dist2, failed.η, label="Failure", color=(:red, 0.5))
    axislegend(ax_dist2)

    # 3. Distribution of skill_factor
    ax_dist3 = Axis(fig_dist[2, 3],
        title = "Distribution of skill_factor",
        xlabel = "skill_factor",
        ylabel = "Density"
    )
    density!(ax_dist3, successful.skill_factor, label="Success", color=(:blue, 0.5))
    density!(ax_dist3, failed.skill_factor, label="Failure", color=(:red, 0.5))
    axislegend(ax_dist3)

    # 4. Distribution of A
    ax_dist4 = Axis(fig_dist[2, 4],
        title = "Distribution of A",
        xlabel = "A",
        ylabel = "Density"
    )
    density!(ax_dist4, successful.A, label="Success", color=(:blue, 0.5))
    density!(ax_dist4, failed.A, label="Failure", color=(:red, 0.5))
    axislegend(ax_dist4)

    # Add heatmap visualizations
    fig_heat = Figure(size=(1800, 1200), fontsize=12)

    # Success Rate: Tax vs Carbon Intensity (improved version)
    ax_heat1 = Axis(fig_heat[1, 1],
        title = "Success Rate: Tax vs Carbon Intensity",
        xlabel = "Tax Rate",
        ylabel = "Carbon Intensity"
    )
    
    success_matrix = zeros(50, 50)
    tax_range = range(minimum(results.τ), maximum(results.τ), length=50)
    carbon_range = range(minimum(results.η), maximum(results.η), length=50)
    
    for (i, tax) in enumerate(tax_range), (j, carbon) in enumerate(carbon_range)
        nearby = findall(
            (abs.(results.τ .- tax) .< 0.1) .&
            (abs.(results.η .- carbon) .< 0.1)
        )
        success_matrix[i, j] = isempty(nearby) ? 0.0 : mean(results.success[nearby])
    end
    
    heatmap!(ax_heat1, tax_range, carbon_range, success_matrix, colormap=:viridis)
    Colorbar(fig_heat[1, 2], colormap=:viridis, label="Success Rate")

    # Success Rate: Skill Factor vs Carbon Intensity
    ax_heat2 = Axis(fig_heat[1, 3],
        title = "Success Rate: Skill Factor vs Carbon Intensity",
        xlabel = "Skill Factor",
        ylabel = "Carbon Intensity"
    )
    
    success_matrix_2 = zeros(50, 50)
    skill_range = range(minimum(results.skill_factor), maximum(results.skill_factor), length=50)
    
    for (i, skill) in enumerate(skill_range), (j, carbon) in enumerate(carbon_range)
        nearby = findall(
            (abs.(results.skill_factor .- skill) .< 0.1) .&
            (abs.(results.η .- carbon) .< 0.1)
        )
        success_matrix_2[i, j] = isempty(nearby) ? 0.0 : mean(results.success[nearby])
    end
    
    heatmap!(ax_heat2, skill_range, carbon_range, success_matrix_2, colormap=:viridis)
    Colorbar(fig_heat[1, 4], colormap=:viridis, label="Success Rate")

    # Generate the productivity controlled heatmaps separately
    fig_productivity = generate_productivity_controlled_heatmaps(results)

    # Save all figures
    save("monte_carlo_heatmaps.png", fig_heat, px_per_unit=2)
    save("monte_carlo_analysis_investment.png", fig, px_per_unit=2)
    save("monte_carlo_analysis_extra_investment.png", fig_extra, px_per_unit=2)
    save("monte_carlo_distributions.png", fig_dist, px_per_unit=2)

    # Add after the existing analysis section:
    critical_points = analyze_failure_modes(results)
    
    # Add an additional visualization for failure modes
    fig2 = Figure(size=(1200, 800))
    
    # 3D scatter plot of failures
    ax3d = Axis3(fig2[1, 1:2],
        title = "Failure Modes in Parameter Space",
        xlabel = "τ*η",
        ylabel = "skill_factor",
        zlabel = "A"
    )
    
    scatter!(ax3d,
        failed.τ_x_η, failed.skill_factor, failed.A,
        color = :red, alpha = 0.3,
        label = "Failures"
    )
    
    # Add success boundary points
    scatter!(ax3d,
        successful.τ_x_η, successful.skill_factor, successful.A,
        color = :blue, alpha = 0.1,
        label = "Successes"
    )
    
    axislegend(ax3d)
    
    # Save additional visualization
    save("failure_modes_analysis.png", fig2, px_per_unit=2)
    
    # Add after critical_points = analyze_failure_modes(results):
    interaction_analysis = analyze_advanced_interactions(results)
    
    return results, fig, fig_extra, fig_dist, fig_heat, fig_productivity, critical_points, interaction_analysis
end

# Run the visualization if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    run_visualization_analysis()
end