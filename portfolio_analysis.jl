
function plot_portfolio_distributions_enhanced(sequence_matrix::Matrix{Float64})
    p1 = scatter3d(
        sequence_matrix[:,1],  # A_eff_0
        sequence_matrix[:,2],  # η_eff_0
        sequence_matrix[:,3],  # cost_0
        title="Period 0 Portfolio Distribution",
        xlabel="A_eff",
        ylabel="η_eff",
        zlabel="Cost",
        marker=:circle,
        markersize=2,
        alpha=0.6,
        legend=false,
        color=:blues,
        camera=(45, 45)
    )

    p2 = scatter3d(
        sequence_matrix[:,4],  # A_eff_1
        sequence_matrix[:,5],  # η_eff_1
        sequence_matrix[:,6],  # cost_1
        title="Period 1 Portfolio Distribution",
        xlabel="A_eff",
        ylabel="η_eff",
        zlabel="Cost",
        marker=:circle,
        markersize=2,
        alpha=0.6,
        legend=false,
        color=:reds,
        camera=(45, 45)
    )

    plot_combined = plot(p1, p2, layout=(1,2), size=(1200,600), dpi=300)
    return plot_combined
end


function analyze_domain_coverage(sequence_matrix::Matrix{Float64}, period::Int)
    # Get column indices for this period (0 or 1)
    A_col = period == 0 ? 1 : 4
    η_col = period == 0 ? 2 : 5
    
    # Sort unique values
    A_values = sort(unique(sequence_matrix[:, A_col]))
    η_values = sort(unique(sequence_matrix[:, η_col]))
    
    # Analyze gaps in A_eff
    A_gaps = Float64[]
    for i in 1:(length(A_values)-1)
        gap = A_values[i+1] - A_values[i]
        if gap > 0.01  # Threshold for considering a gap significant
            push!(A_gaps, gap)
        end
    end
    
    # Analyze gaps in η_eff
    η_gaps = Float64[]
    for i in 1:(length(η_values)-1)
        gap = η_values[i+1] - η_values[i]
        if gap > 0.01  # Threshold for considering a gap significant
            push!(η_gaps, gap)
        end
    end
    
    println("\nPeriod $period Domain Analysis:")
    println("A_eff range: [$(minimum(A_values)), $(maximum(A_values))]")
    println("Number of unique A_eff values: $(length(A_values))")
    if !isempty(A_gaps)
        println("Found $(length(A_gaps)) gaps in A_eff > 0.01:")
        println("Gap sizes: $A_gaps")
        println("Locations: $([(A_values[i], A_values[i+1]) for i in 1:(length(A_values)-1) if A_values[i+1] - A_values[i] > 0.01])")
    else
        println("No significant gaps found in A_eff")
    end
    
    println("\nη_eff range: [$(minimum(η_values)), $(maximum(η_values))]")
    println("Number of unique η_eff values: $(length(η_values))")
    if !isempty(η_gaps)
        println("Found $(length(η_gaps)) gaps in η_eff > 0.01:")
        println("Gap sizes: $η_gaps")
        println("Locations: $([(η_values[i], η_values[i+1]) for i in 1:(length(η_values)-1) if η_values[i+1] - η_values[i] > 0.01])")
    else
        println("No significant gaps found in η_eff")
    end
    
    # Create scatter plot of the domain
    p = scatter(
        sequence_matrix[:, A_col],
        sequence_matrix[:, η_col],
        title="Period $period Portfolio Distribution",
        xlabel="A_eff",
        ylabel="η_eff",
        marker=:circle,
        markersize=2,
        alpha=0.6,
        legend=false,
        color=:blues
    )
    
    return p
end