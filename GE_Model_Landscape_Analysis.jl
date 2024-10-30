using Distributions
using LinearAlgebra
using Statistics
using Plots
using DataFrames
using ProgressMeter
using Random
using CSV
include("GE_Model_core_for_functional_analysis.jl")

"""
Represents a point in the parameter space with its evaluation results
"""
struct LandscapePoint
    params::Dict{String, Float64}
    result::Dict{String, Any}
    success::Bool
    objective_value::Union{Float64, Nothing}
end

"""
Generate Latin Hypercube samples for parameter space exploration
"""
function generate_parameter_samples(n_samples::Int)
    # Define parameter ranges with restricted credibility
    param_ranges = Dict(
        "τ_current" => (0.0, 0.5),
        "τ_announced" => (0.0, 0.5),
        "η_mean" => (0.5, 1.5),
        "η_std" => (0.1, 0.3),
        "credibility" => (0.8, 1.0),
        "μ_A" => (0.8, 1.2),        # Mean productivity
        "σ_A" => (0.1, 0.3),        # Productivity variation
        "γ" => (0.05, 0.15)         # Adjustment cost parameter
    )
    
    # Generate Latin Hypercube samples
    n_params = length(param_ranges)
    lhs_samples = reduce(hcat, [range(0, 1, length=n_samples) for _ in 1:n_params])
    for col in 1:n_params
        lhs_samples[:, col] = lhs_samples[randperm(n_samples), col]
    end
    
    # Transform to parameter ranges
    param_names = collect(keys(param_ranges))
    samples = Vector{Dict{String, Float64}}(undef, n_samples)
    
    for i in 1:n_samples
        samples[i] = Dict(
            name => param_ranges[name][1] + 
                   (param_ranges[name][2] - param_ranges[name][1]) * lhs_samples[i, j]
            for (j, name) in enumerate(param_names)
        )
    end
    
    return samples
end

"""
Evaluate model at a single point in parameter space
"""
function evaluate_point(params_dict::Dict{String, Float64})
    # Create modified parameters with new values
    modified_params = ModelParameters(
        μ_A = params_dict["μ_A"],
        σ_A = params_dict["σ_A"],
        γ = params_dict["γ"],
    )
    
    expectations = form_tax_expectations(
        params_dict["τ_current"],
        params_dict["τ_announced"],
        params_dict["η_mean"],
        params_dict["η_std"],
        params_dict["credibility"]
    )
    
    result = compute_equilibrium(expectations, modified_params)
    success = !get(result, "error", false)
    
    # Calculate a pseudo-objective value for landscape analysis
    # Using economic metrics that should be maximized
    objective_value = if success
        result["Y_t"] * (1 - result["η_t"]) - 
        0.5 * (result["L_t"])^2  # Simple welfare approximation
    else
        nothing
    end
    
    return LandscapePoint(params_dict, result, success, objective_value)
end

"""
Analyze local geometry around a point
"""
function analyze_local_geometry(point::LandscapePoint, nearby_points::Vector{LandscapePoint})
    if !point.success || isempty(nearby_points)
        return Dict{String, Any}()
    end
    
    # Calculate gradients for key metrics
    metrics = ["Y_t", "L_t", "η_t"]
    gradients = Dict{String, Vector{Float64}}()
    
    for metric in metrics
        if haskey(point.result, metric)
            grad = zeros(8)  # Updated for 8 parameters
            center_val = point.result[metric]
            
            for nearby in nearby_points
                if nearby.success
                    param_diff = [
                        nearby.params[p] - point.params[p] 
                        for p in ["τ_current", "τ_announced", "η_mean", "η_std", 
                                "credibility", "μ_A", "σ_A", "γ"]
                    ]
                    metric_diff = nearby.result[metric] - center_val
                    grad .+= (metric_diff .* param_diff) ./ (norm(param_diff)^2 + 1e-10)
                end
            end
            
            gradients[metric] = grad ./ length(nearby_points)
        end
    end
    
    return Dict(
        "gradients" => gradients,
        "success_rate" => mean(p.success for p in nearby_points),
        "objective_variation" => std(filter(!isnothing, [p.objective_value for p in nearby_points]))
    )
end

"""
Analyze convexity and smoothness of the objective function
"""
function analyze_function_shape(points::Vector{LandscapePoint})
    successful_points = filter(p -> p.success, points)
    
    # Extract parameter values and objective values
    param_matrix = zeros(length(successful_points), 5)
    obj_values = zeros(length(successful_points))
    
    for (i, point) in enumerate(successful_points)
        param_matrix[i, :] = [point.params[p] for p in 
            ["τ_current", "τ_announced", "η_mean", "η_std", "credibility"]]
        obj_values[i] = point.objective_value
    end
    
    # Analyze local convexity by checking midpoints
    n_tests = min(1000, length(successful_points) ÷ 2)
    convexity_violations = 0
    smoothness_measure = 0.0
    
    for _ in 1:n_tests
        # Randomly select two points
        idx1, idx2 = rand(1:length(successful_points), 2)
        p1, p2 = successful_points[idx1], successful_points[idx2]
        
        # Check midpoint
        mid_params = Dict(
            k => (p1.params[k] + p2.params[k])/2 
            for k in keys(p1.params)
        )
        
        mid_point = evaluate_point(mid_params)
        
        if mid_point.success && !isnothing(mid_point.objective_value)
            # Test convexity
            if mid_point.objective_value < (p1.objective_value + p2.objective_value)/2
                convexity_violations += 1
            end
            
            # Measure smoothness (using second differences)
            smoothness_measure += abs(2*mid_point.objective_value - 
                                   p1.objective_value - p2.objective_value)
        end
    end
    
    return Dict(
        "convexity_violation_rate" => convexity_violations/n_tests,
        "average_smoothness" => smoothness_measure/n_tests,
        "success_count" => length(successful_points)
    )
end

"""
Analyze parameter relationships and sensitivity
"""
function analyze_parameter_relationships(points::Vector{LandscapePoint})
    successful_points = filter(p -> p.success, points)
    
    # Create arrays for correlation analysis
    params = ["τ_current", "τ_announced", "η_mean", "η_std", "credibility"]
    param_values = Dict(p => Float64[] for p in params)
    objective_values = Float64[]
    
    for point in successful_points
        for p in params
            push!(param_values[p], point.params[p])
        end
        push!(objective_values, point.objective_value)
    end
    
    # Calculate correlations with objective
    correlations = Dict(
        p => cor(param_values[p], objective_values)
        for p in params
    )
    
    # Calculate parameter interactions (correlation matrix)
    param_interactions = zeros(length(params), length(params))
    for (i, p1) in enumerate(params)
        for (j, p2) in enumerate(params)
            param_interactions[i,j] = cor(param_values[p1], param_values[p2])
        end
    end
    
    return Dict(
        "objective_correlations" => correlations,
        "parameter_interactions" => param_interactions,
        "parameters" => params
    )
end

"""
Store analysis results in CSV files
"""
function save_analysis_results(points::Vector{LandscapePoint}, 
                             shape_analysis::Dict, 
                             relationship_analysis::Dict,
                             output_dir::String="analysis_results")
    mkpath(output_dir)
    
    # 1. Main results dataframe - updated with new parameters and proper missing value handling
    main_data = DataFrame(
        τ_current = Float64[],
        τ_announced = Float64[],
        η_mean = Float64[],
        η_std = Float64[],
        credibility = Float64[],
        μ_A = Float64[],
        σ_A = Float64[],
        γ = Float64[],
        success = Bool[],
        objective_value = Union{Float64, Missing}[],
        Y_t = Union{Float64, Missing}[],
        L_t = Union{Float64, Missing}[],
        η_t = Union{Float64, Missing}[],
        Technology_Split = Union{Float64, Missing}[],
        Labor_Efficiency = Union{Float64, Missing}[]
    )
    
    for point in points
        row = Dict(
            :τ_current => point.params["τ_current"],
            :τ_announced => point.params["τ_announced"],
            :η_mean => point.params["η_mean"],
            :η_std => point.params["η_std"],
            :credibility => point.params["credibility"],
            :μ_A => point.params["μ_A"],
            :σ_A => point.params["σ_A"],
            :γ => point.params["γ"],
            :success => point.success,
            :objective_value => something(point.objective_value, missing),  # Convert nothing to missing
            :Y_t => point.success ? point.result["Y_t"] : missing,
            :L_t => point.success ? point.result["L_t"] : missing,
            :η_t => point.success ? point.result["η_t"] : missing,
            :Technology_Split => point.success ? point.result["Technology_Split"] : missing,
            :Labor_Efficiency => point.success ? point.result["Labor_Efficiency"] : missing
        )
    
        push!(main_data, row)
    end
    
    # 2. Parameter bins analysis - updated with new parameters
    param_bins = Dict()
    for param in ["τ_current", "τ_announced", "η_mean", "η_std", "credibility", 
                 "μ_A", "σ_A", "γ"]  # Updated parameter list
        values = [p.params[param] for p in points]
        bins = range(minimum(values), maximum(values), length=10)
        success_rates = Float64[]
        avg_objectives = Float64[]
        
        for i in 1:(length(bins)-1)
            mask = (values .>= bins[i]) .& (values .< bins[i+1])
            points_in_bin = points[mask]
            
            push!(success_rates, mean(p.success for p in points_in_bin))
            successful_objectives = filter(!isnothing, [p.objective_value for p in points_in_bin if p.success])
            push!(avg_objectives, isempty(successful_objectives) ? NaN : mean(successful_objectives))
        end
        
        param_bins[param] = DataFrame(
            bin_start = bins[1:end-1],
            bin_end = bins[2:end],
            success_rate = success_rates,
            avg_objective = avg_objectives
        )
    end
    
    # 3. Feasibility analysis - updated with new parameters
    feasibility_data = DataFrame(
        parameter = String[],
        feasible_min = Float64[],
        feasible_max = Float64[],
        optimal_range_min = Float64[],
        optimal_range_max = Float64[]
    )
    
    successful_points = filter(p -> p.success, points)
    if !isempty(successful_points)
        obj_values = [p.objective_value for p in successful_points]
        top_quartile = quantile(filter(!isnothing, obj_values), 0.75)
        
        for param in ["τ_current", "τ_announced", "η_mean", "η_std", "credibility",
                     "μ_A", "σ_A", "γ"]  # Updated parameter list
            feasible_values = [p.params[param] for p in successful_points]
            optimal_points = successful_points[obj_values .>= top_quartile]
            optimal_values = [p.params[param] for p in optimal_points]
            
            push!(feasibility_data, Dict(
                :parameter => param,
                :feasible_min => minimum(feasible_values),
                :feasible_max => maximum(feasible_values),
                :optimal_range_min => minimum(optimal_values),
                :optimal_range_max => maximum(optimal_values)
            ))
        end
    end
    
    # 4. Shape analysis results
    shape_df = DataFrame(
        metric = String[],
        value = Float64[]
    )
    
    push!(shape_df, Dict(
        :metric => "convexity_violation_rate",
        :value => shape_analysis["convexity_violation_rate"]
    ))
    push!(shape_df, Dict(
        :metric => "average_smoothness",
        :value => shape_analysis["average_smoothness"]
    ))
    
    # Save all dataframes - no need for transform now as we're using missing
    CSV.write(joinpath(output_dir, "main_results.csv"), main_data)
    CSV.write(joinpath(output_dir, "feasibility_analysis.csv"), feasibility_data)
    CSV.write(joinpath(output_dir, "shape_analysis.csv"), shape_df)
    
    for (param, df) in param_bins
        CSV.write(joinpath(output_dir, "param_bins_$(param).csv"), df)
    end
    
    return main_data, feasibility_data, shape_df, param_bins
end

"""
Enhanced analyze_model_landscape function
"""
function analyze_model_landscape(n_samples::Int=1000)
    println("Generating parameter samples...")
    samples = generate_parameter_samples(n_samples)
    
    println("Evaluating model across parameter space...")
    points = Vector{LandscapePoint}(undef, n_samples)
    
    p = Progress(n_samples)
    Threads.@threads for i in 1:n_samples
        points[i] = evaluate_point(samples[i])
        next!(p)
    end
    
    # Analyze results
    success_rate = mean(p.success for p in points)
    println("\nOverall success rate: $(round(success_rate * 100, digits=2))%")
    
    # Find regions of interest
    successful_points = filter(p -> p.success, points)
    if !isempty(successful_points)
        obj_values = filter(!isnothing, [p.objective_value for p in successful_points])
        best_point = successful_points[argmax(obj_values)]
        
        println("\nBest point found:")
        for (param, value) in best_point.params
            println("$param: $(round(value, digits=3))")
        end
        
        # Local geometry analysis around best point
        k_nearest = 10
        distances = [sum((collect(values(p.params)) .- collect(values(best_point.params))).^2) 
                    for p in points]
        nearby_indices = partialsort(1:length(points), 1:k_nearest, by=i->distances[i])
        nearby_points = points[nearby_indices]
        
        local_geometry = analyze_local_geometry(best_point, nearby_points)
        
        println("\nLocal geometry around best point:")
        if haskey(local_geometry, "gradients")
            for (metric, grad) in local_geometry["gradients"]
                println("$metric gradient magnitude: $(round(norm(grad), digits=3))")
            end
        end
    end
    
    println("Analyzing function shape...")
    shape_analysis = analyze_function_shape(points)
    println("\nFunction shape analysis:")
    println("Convexity violation rate: $(round(shape_analysis["convexity_violation_rate"] * 100, digits=2))%")
    println("Average smoothness measure: $(round(shape_analysis["average_smoothness"], digits=4))")
    
    println("\nAnalyzing parameter relationships...")
    relationship_analysis = analyze_parameter_relationships(points)
    
    println("\nParameter correlations with objective:")
    for (param, corr) in relationship_analysis["objective_correlations"]
        println("$param: $(round(corr, digits=3))")
    end
    
    println("\nStrong parameter interactions (|correlation| > 0.3):")
    params = relationship_analysis["parameters"]
    for i in 1:length(params)
        for j in (i+1):length(params)
            corr = relationship_analysis["parameter_interactions"][i,j]
            if abs(corr) > 0.3
                println("$(params[i]) - $(params[j]): $(round(corr, digits=3))")
            end
        end
    end
    
    # Save all results
    println("\nSaving analysis results...")
    main_data, feasibility_data, shape_df, param_bins = 
        save_analysis_results(points, shape_analysis, relationship_analysis)
    
    # Additional summary statistics
    println("\nFeasibility ranges for optimal solutions:")
    for row in eachrow(feasibility_data)
        println("$(row.parameter):")
        println("  Feasible range: [$(round(row.feasible_min, digits=3)), $(round(row.feasible_max, digits=3))]")
        println("  Optimal range: [$(round(row.optimal_range_min, digits=3)), $(round(row.optimal_range_max, digits=3))]")
    end
    
    return Dict(
        "points" => points,
        "shape_analysis" => shape_analysis,
        "relationship_analysis" => relationship_analysis,
        "main_data" => main_data,
        "feasibility_data" => feasibility_data,
        "shape_df" => shape_df,
        "param_bins" => param_bins
    )
end

# Run analysis
results = analyze_model_landscape(100000)