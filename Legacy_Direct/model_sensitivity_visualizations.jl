using CairoMakie
using DataFrames
using CSV
using Statistics
using StatsBase

function load_and_prepare_data()
    # Load results from the main sensitivity analysis
    results = CSV.read("monte_carlo_results_investment.csv", DataFrame)
    successful = results[results.success .== true, :]
    failed = results[results.success .== false, :]
    return results, successful, failed
end

function plot_labor_efficiency_analysis(successful)
    fig = Figure(size=(1800, 1200))

    # 1. Labor Efficiency vs Technology Change
    ax1 = Axis(fig[1, 1],
        title = "Labor Efficiency vs Technology Change",
        xlabel = "Δη",
        ylabel = "Labor Efficiency"
    )
    
    # Calculate Δη from technology split
    Δη = successful.technology_split .* (successful.η .- successful.η)
    scatter!(ax1, Δη, successful.labor_efficiency)

    # 2. Labor Efficiency Distribution by Skill Factor
    ax2 = Axis(fig[1, 2],
        title = "Labor Efficiency Distribution by Skill Factor",
        xlabel = "Skill Factor",
        ylabel = "Labor Efficiency"
    )
    
    # Create boxplot groups based on skill factor quantiles
    skill_groups = cut(successful.skill_factor, 6)
    boxplot!(ax2, skill_groups, successful.labor_efficiency)

    # 3. Average Labor Efficiency (A₀ vs Skill Factor)
    ax3 = Axis(fig[2, 1],
        title = "Average Labor Efficiency (A₀ vs Skill Factor)",
        xlabel = "Skill Factor",
        ylabel = "A₀"
    )
    
    labor_eff_matrix = zeros(50, 50)
    x_range = range(minimum(successful.skill_factor), maximum(successful.skill_factor), length=50)
    y_range = range(minimum(successful.A), maximum(successful.A), length=50)
    
    for (i, x) in enumerate(x_range), (j, y) in enumerate(y_range)
        nearby = findall(abs.(successful.skill_factor .- x) .< 0.1 .&
                        abs.(successful.A .- y) .< 0.1)
        labor_eff_matrix[i, j] = mean(successful.labor_efficiency[nearby])
    end
    
    heatmap!(ax3, x_range, y_range, labor_eff_matrix)
    Colorbar(fig[2, 2], colormap=:viridis, label="Labor Efficiency")

    # 4. Labor Efficiency vs Output
    ax4 = Axis(fig[2, 3],
        title = "Labor Efficiency vs Output",
        xlabel = "Output (Y₀)",
        ylabel = "Labor Efficiency"
    )
    
    scatter!(ax4, successful.effective_output, successful.labor_efficiency,
            color=successful.skill_factor, colormap=:viridis)
    Colorbar(fig[2, 4], colormap=:viridis, label="Skill Factor")

    save("labor_efficiency_analysis.png", fig, px_per_unit=2)
    return fig
end

function plot_success_rate_analysis(results)
    fig = Figure(size=(1800, 900))

    # 1. Success Rate: Tax vs Carbon Intensity
    ax1 = Axis(fig[1, 1],
        title = "Success Rate: Tax vs Carbon Intensity",
        xlabel = "Tax Rate",
        ylabel = "Carbon Intensity"
    )
    
    success_matrix = zeros(50, 50)
    tax_range = range(minimum(results.τ), maximum(results.τ), length=50)
    carbon_range = range(minimum(results.η), maximum(results.η), length=50)
    
    for (i, tax) in enumerate(tax_range), (j, carbon) in enumerate(carbon_range)
        nearby = findall(abs.(results.τ .- tax) .< 0.1 .&
                        abs.(results.η .- carbon) .< 0.1)
        success_matrix[i, j] = mean(results.success[nearby])
    end
    
    heatmap!(ax1, tax_range, carbon_range, success_matrix)
    Colorbar(fig[1, 2], colormap=:viridis, label="Success Rate")

    # 2. Success Rate: Skill Factor vs Carbon Intensity
    ax2 = Axis(fig[1, 3],
        title = "Success Rate: Skill Factor vs Carbon Intensity",
        xlabel = "Skill Factor",
        ylabel = "Carbon Intensity"
    )
    
    success_matrix_2 = zeros(50, 50)
    skill_range = range(minimum(results.skill_factor), maximum(results.skill_factor), length=50)
    
    for (i, skill) in enumerate(skill_range), (j, carbon) in enumerate(carbon_range)
        nearby = findall(abs.(results.skill_factor .- skill) .< 0.1 .&
                        abs.(results.η .- carbon) .< 0.1)
        success_matrix_2[i, j] = mean(results.success[nearby])
    end
    
    heatmap!(ax2, skill_range, carbon_range, success_matrix_2)
    Colorbar(fig[1, 4], colormap=:viridis, label="Success Rate")

    save("success_rate_analysis.png", fig, px_per_unit=2)
    return fig
end

function plot_output_analysis(successful)
    fig = Figure(size=(1800, 900))

    # 1. Effective Output vs τ*η
    ax1 = Axis(fig[1, 1],
        title = "Effective Output vs τ*η",
        xlabel = "τ*η",
        ylabel = "Effective Output"
    )
    
    scatter!(ax1, successful.τ_x_η, successful.effective_output)

    # 2. Distribution of Output Multiplier
    ax2 = Axis(fig[1, 2],
        title = "Distribution of Output Multiplier (1 - τ*η)",
        xlabel = "Output Multiplier",
        ylabel = "Count"
    )
    
    hist!(ax2, successful.output_multiplier, bins=30)

    save("output_analysis.png", fig, px_per_unit=2)
    return fig
end

function main()
    results, successful, failed = load_and_prepare_data()
    
    # Generate all visualizations
    labor_fig = plot_labor_efficiency_analysis(successful)
    success_fig = plot_success_rate_analysis(results)
    output_fig = plot_output_analysis(successful)
    
    # Calculate and print labor adjustment cost effects
    println("\nLabor Adjustment Cost Analysis:")
    
    # Calculate correlation between γ and labor efficiency
    γ_efficiency_cor = cor(successful.γ, successful.labor_efficiency)
    println("Correlation between adjustment cost (γ) and labor efficiency: ", 
            round(γ_efficiency_cor, digits=3))
    
    # Calculate average labor efficiency for different γ levels
    γ_quantiles = quantile(successful.γ, [0.25, 0.5, 0.75])
    for (i, q) in enumerate(γ_quantiles)
        group = successful[successful.γ .<= q, :]
        println("Average labor efficiency for γ ≤ $(round(q, digits=3)): ",
                round(mean(group.labor_efficiency), digits=3))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 