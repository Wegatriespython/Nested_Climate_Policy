using DataFrames
using CSV
using Plots
using StatsPlots
using Statistics
using LinearAlgebra
using KernelDensity

# Add at the start of the script, after imports:
const OUTPUT_DIR = "V:\\Approach 2\\Output\\Objective Domain"
mkpath(OUTPUT_DIR)  # Create directory if it doesn't exist

"""
Load and preprocess all analysis results
"""
function load_analysis_data(results_dir::String="analysis_results")
    main_data = CSV.read(joinpath(results_dir, "main_results.csv"), DataFrame)
    shape_data = CSV.read(joinpath(results_dir, "shape_analysis.csv"), DataFrame)
    feasibility_data = CSV.read(joinpath(results_dir, "feasibility_analysis.csv"), DataFrame)
    
    # Load parameter bins - updated parameter list
    param_bins = Dict()
    for param in ["τ_current", "τ_announced", "η_mean", "η_std", "credibility", 
                 "μ_A", "σ_A", "γ"]
        file = joinpath(results_dir, "param_bins_$(param).csv")
        if isfile(file)
            param_bins[param] = CSV.read(file, DataFrame)
        end
    end
    
    return main_data, shape_data, feasibility_data, param_bins
end

"""
Create objective function landscape visualizations
"""
function plot_objective_landscape(main_data::DataFrame)
    successful = main_data[.!ismissing.(main_data.objective_value), :]
    params = ["τ_current", "τ_announced", "η_mean", "η_std", "credibility", 
             "μ_A", "σ_A", "γ"]
    
    # Group parameters for more organized plotting
    param_groups = [
        ("Policy", ["τ_current", "τ_announced", "credibility"]),
        ("Technology", ["η_mean", "η_std"]),
        ("Production", ["μ_A", "σ_A", "γ"])
    ]
    
    # Create plots within each group and between groups
    for (group1_name, group1_params) in param_groups
        for (group2_name, group2_params) in param_groups
            for param1 in group1_params
                for param2 in group2_params
                    if param1 < param2  # Avoid duplicate plots
                        p = scatter(
                            successful[:, param1], 
                            successful[:, param2], 
                            zcolor=successful.objective_value,
                            marker_z=successful.objective_value,
                            title="$group1_name vs $group2_name\n$param1 vs $param2",
                            xlabel=param1,
                            ylabel=param2,
                            colorbar_title="Objective",
                            markersize=4,
                            alpha=0.6,
                            size=(800, 600)
                        )
                        savefig(p, joinpath(OUTPUT_DIR, "objective_landscape_$(param1)_vs_$(param2).png"))
                    end
                end
            end
        end
    end
end

"""
Analyze and visualize non-convexity patterns
"""
function analyze_nonconvexity(main_data::DataFrame)
    successful = main_data[.!ismissing.(main_data.objective_value), :]
    params = ["τ_current", "τ_announced", "η_mean", "η_std", "credibility", 
             "μ_A", "σ_A", "γ"]
    
    # Group parameters as before
    param_groups = [
        ("Policy", ["τ_current", "τ_announced", "credibility"]),
        ("Technology", ["η_mean", "η_std"]),
        ("Production", ["μ_A", "σ_A", "γ"])
    ]
    
    # Create convexity plots within and between groups
    for (group1_name, group1_params) in param_groups
        for (group2_name, group2_params) in param_groups
            for param1 in group1_params
                for param2 in group2_params
                    if param1 < param2
                        x = successful[:, param1]
                        y = successful[:, param2]
                        z = successful.objective_value
                        
                        # Estimate local non-convexity
                        grid_size = 20
                        x_bins = range(minimum(x), maximum(x), length=grid_size)
                        y_bins = range(minimum(y), maximum(y), length=grid_size)
                        convexity_violations = zeros(grid_size-1, grid_size-1)
                        
                        for xi in 1:(grid_size-1), yi in 1:(grid_size-1)
                            mask = (x .>= x_bins[xi]) .& (x .< x_bins[xi+1]) .&
                                   (y .>= y_bins[yi]) .& (y .< y_bins[yi+1])
                            local_points = z[mask]
                            
                            if length(local_points) > 2
                                midpoint = mean(local_points)
                                endpoints = [minimum(local_points), maximum(local_points)]
                                convexity_violations[xi, yi] = 
                                    midpoint < (endpoints[1] + endpoints[2])/2 ? 1.0 : 0.0
                            end
                        end
                        
                        p = heatmap(x_bins[1:end-1], y_bins[1:end-1], convexity_violations',
                                   title="Non-convexity: $group1_name vs $group2_name\n$param1 vs $param2",
                                   xlabel=param1,
                                   ylabel=param2,
                                   color=:viridis,
                                   size=(800, 600),
                                   colorbar_title="Convexity Violation Rate")
                        
                        savefig(p, joinpath(OUTPUT_DIR, "convexity_analysis_$(param1)_vs_$(param2).png"))
                    end
                end
            end
        end
    end
    
    # Create density plots for each parameter group
    density_plots = []
    for (group_name, group_params) in param_groups
        # Filter out extreme values using quantiles
        filtered_data = successful[successful.objective_value .< quantile(successful.objective_value, 0.99), :]
        
        # Create bins for the first parameter in the group
        param = group_params[1]
        n_bins = 5  # Reduce number of bins to avoid sparsity
        bin_edges = range(minimum(filtered_data[:, param]), 
                         maximum(filtered_data[:, param]), 
                         length=n_bins+1)
        
        # Initialize bin labels with a default value
        bin_labels = fill("undefined", size(filtered_data, 1))
        
        # Assign bin labels and ensure every point gets a label
        for i in 1:size(filtered_data, 1)
            value = filtered_data[i, param]
            for j in 1:n_bins
                if j == n_bins && value >= bin_edges[j] && value <= bin_edges[j+1]
                    bin_labels[i] = "$(round(bin_edges[j], digits=2)) - $(round(bin_edges[j+1], digits=2))"
                elseif value >= bin_edges[j] && value < bin_edges[j+1]
                    bin_labels[i] = "$(round(bin_edges[j], digits=2)) - $(round(bin_edges[j+1], digits=2))"
                end
            end
        end
        
        # Verify all points have valid labels
        if any(label -> label == "undefined", bin_labels)
            @warn "Some points were not assigned to bins for parameter $param"
            continue  # Skip this parameter if there are undefined labels
        end
        
        try
            p = density(filtered_data.objective_value,
                       group=bin_labels,
                       title="Objective Distribution by $group_name\nGrouped by $param",
                       xlabel="Objective Value",
                       ylabel="Density",
                       size=(800, 600),
                       legend=:topright)
            push!(density_plots, p)
            savefig(p, joinpath(OUTPUT_DIR, "objective_density_$(group_name).png"))
        catch e
            @warn "Failed to create density plot for $group_name" exception=e
        end
    end
    
    return density_plots
end

"""
Analyze parameter sensitivity and interactions
"""
function analyze_parameter_sensitivity(main_data::DataFrame)
    successful = main_data[.!ismissing.(main_data.objective_value), :]
    params = ["τ_current", "τ_announced", "η_mean", "η_std", "credibility", 
             "μ_A", "σ_A", "γ"]
    
    # 1. Create correlation matrix plot with grouped parameters
    cor_matrix = cor(Matrix(successful[:, params]))
    correlation_plot = heatmap(params, params, cor_matrix,
                             title="Parameter Correlations",
                             color=:seismic,
                             clim=(-1, 1),
                             size=(1000, 800),
                             xticks=(1:length(params), params),
                             yticks=(1:length(params), params),
                             xrotation=45)
    savefig(correlation_plot, joinpath(OUTPUT_DIR, "parameter_correlations.png"))
    
    # 2. Create success rate analysis with parameter grouping
    param_groups = [
        ("Policy", ["τ_current", "τ_announced", "credibility"]),
        ("Technology", ["η_mean", "η_std"]),
        ("Production", ["μ_A", "σ_A", "γ"])
    ]
    
    for (group_name, group_params) in param_groups
        plots = []
        for param in group_params
            edges = range(minimum(main_data[:, param]), maximum(main_data[:, param]), length=20)
            centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
            rates = Float64[]
            
            for i in 1:(length(edges)-1)
                mask = (main_data[:, param] .>= edges[i]) .& (main_data[:, param] .< edges[i+1])
                push!(rates, mean(.!ismissing.(main_data[mask, :objective_value])))
            end
            
            # Enhanced plot with clearer labeling
            param_labels = Dict(
                "τ_current" => "Current Tax",
                "τ_announced" => "Announced Tax",
                "credibility" => "Credibility",
                "η_mean" => "Mean Carbon Intensity",
                "η_std" => "Carbon Intensity Std",
                "μ_A" => "Mean Productivity",
                "σ_A" => "Productivity Std",
                "γ" => "Adjustment Cost"
            )
            
            p = plot(centers, rates,
                    title=param_labels[param],  # Use full parameter name in title
                    xlabel=param,               # Keep technical parameter name on x-axis
                    ylabel="Success Rate",
                    legend=false,
                    size=(800, 600),
                    linewidth=2,
                    grid=true,                 # Add grid for better readability
                    fontsize=12)               # Increase font size
            push!(plots, p)
        end
        
        # Create combined plot with clear group title
        n_params = length(group_params)
        combined_plot = plot(plots..., 
                           layout=(1, n_params), 
                           size=(800*n_params, 600),
                           title="$group_name Parameter Success Rates",
                           titlefontsize=16,
                           margin=10Plots.mm)
        savefig(combined_plot, joinpath(OUTPUT_DIR, "success_rates_$(group_name).png"))
    end
    
    return correlation_plot
end

# Main analysis
println("Loading data...")
main_data, shape_data, feasibility_data, param_bins = load_analysis_data()

println("Creating objective landscape plots...")
plot_objective_landscape(main_data)

println("Analyzing non-convexity...")
density_plots = analyze_nonconvexity(main_data)

println("Analyzing parameter sensitivity...")
correlation_plot = analyze_parameter_sensitivity(main_data)

println("Analysis complete. Check the generated PNG files in: $OUTPUT_DIR")