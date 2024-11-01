# VintageCapitalMatrix.jl

using LinearAlgebra
using Parameters
using DataStructures
using Plots
using Statistics
using Distributions

"""
Core data structures for the vintage capital model with within-vintage variation
"""
@with_kw struct TechnologySet
    A::Vector{Float64}      # Vector of productivity values
    η::Vector{Float64}      # Vector of emissions intensity values
    
    function TechnologySet(A::Vector{Float64}, η::Vector{Float64})
        @assert length(A) == length(η) "Productivity and emissions vectors must have the same length"
        new(A, η)
    end
end

@with_kw struct OperatedCapital
    weights::Matrix{Float64}          # Matrix of weights over existing capital (each row is a combination)
    base_tech::TechnologySet          # Original technology characteristics
    operation_mode::TechnologySet     # Operation modes with newer tech
end

@with_kw struct NewInvestment
    weights::Matrix{Float64}          # Matrix of weights over new technologies (each row is a combination)
    technologies::TechnologySet       # New technology characteristics
end

@with_kw struct VintagePortfolio
    existing_capital::OperatedCapital
    new_investment::NewInvestment
end

"""
Generate weight combinations using Dirichlet sampling
"""
function generate_weight_matrix_dirichlet(
    n::Int, 
    n_samples::Int=100; 
    α::Float64=1.0  # Concentration parameter
)::Matrix{Float64}
    if n < 1
        error("Number of technologies must be at least 1.")
    end
    if n == 1
        return reshape([1.0], 1, 1)
    end
    
    # Create Dirichlet distribution with specified concentration
    dist = Dirichlet(n, α)
    
    # Generate samples and convert to matrix
    samples = reduce(hcat, rand(dist, n_samples))'
    
    return samples
end

"""
Generate Vintage Portfolios using Dirichlet sampling
"""
function generate_vintage_portfolios_dirichlet(
    tech_minus1::TechnologySet,
    tech0::TechnologySet;
    n_samples::Int=50,  # Samples per split
    n_splits::Int=10    # Number of splits between old and new tech
)::Vector{VintagePortfolio}
    portfolios = Vector{VintagePortfolio}()
    
    # Generate operation modes
    operated_tech = generate_operation_modes(tech_minus1, tech0, n_modes=2)
    
    n_operated = length(operated_tech.A)
    n_new = length(tech0.A)
    
    println("Debug: Generating samples for $n_operated operated modes and $n_new new technologies")
    
    # Generate splits
    splits = LinRange(0, 1, n_splits)
    
    for split in splits
        # Generate weights using Dirichlet sampling
        operated_weights = generate_weight_matrix_dirichlet(n_operated, n_samples)
        new_weights = generate_weight_matrix_dirichlet(n_new, n_samples)
        
        # Scale weights by split
        op_weight_scaled = operated_weights .* split
        new_weight_scaled = new_weights .* (1 - split)
        
        # Create portfolios from samples
        for i in 1:n_samples
            # Create normalized weights
            total_weight = sum(op_weight_scaled[i,:]) + sum(new_weight_scaled[i,:])
            if total_weight ≈ 0
                continue
            end
            
            normalized_op = reshape(op_weight_scaled[i,:], 1, :) ./ total_weight
            normalized_new = reshape(new_weight_scaled[i,:], 1, :) ./ total_weight
            
            # Create portfolio
            portfolio = VintagePortfolio(
                existing_capital=OperatedCapital(
                    weights=normalized_op,
                    base_tech=tech_minus1,
                    operation_mode=operated_tech
                ),
                new_investment=NewInvestment(
                    weights=normalized_new,
                    technologies=tech0
                )
            )
            push!(portfolios, portfolio)
        end
    end
    
    println("Debug: Total portfolios generated: $(length(portfolios))")
    return portfolios
end

"""
Generate operation modes for a given technology set
"""
function generate_operation_modes(
    old_tech::TechnologySet,
    new_tech::TechnologySet;
    n_modes::Int=3  # Number of operation modes per technology
)::TechnologySet
    n_old = length(old_tech.A)
    
    # Generate mode factors evenly spaced between 0 and 1
    mode_factors = LinRange(0, 1, n_modes)
    A_new = maximum(new_tech.A)
    η_new = minimum(new_tech.η)
    
    A_composite = Float64[]
    η_composite = Float64[]
    
    for i in 1:n_old
        for mode in mode_factors
            # Compute weighted combination of old and new technology characteristics
            A_mode = old_tech.A[i] * (1 - mode) + A_new * mode
            η_mode = old_tech.η[i] * (1 - mode) + η_new * mode
            push!(A_composite, A_mode)
            push!(η_composite, η_mode)
        end
    end
    
    println("Debug: Generated $(length(A_composite)) operation modes")
    return TechnologySet(A_composite, η_composite)
end

"""
Calculate effective characteristics for a single portfolio
"""
function compute_effective_characteristics(p::VintagePortfolio)::Tuple{Float64, Float64}
    # Existing capital contribution
    A_existing = sum(p.existing_capital.weights .* p.existing_capital.operation_mode.A)
    η_existing = sum(p.existing_capital.weights .* p.existing_capital.operation_mode.η)
    
    # New investment contribution
    A_new = sum(p.new_investment.weights .* p.new_investment.technologies.A)
    η_new = sum(p.new_investment.weights .* p.new_investment.technologies.η)
    
    # Total effective characteristics
    A_eff = A_existing + A_new
    η_eff = η_existing + η_new
    
    return (A_eff, η_eff)
end

"""
Calculate effective characteristics using matrix multiplication
Each row in weights_matrix corresponds to a unique combination of weights
"""
function compute_effective_characteristics_matrix(weights_matrix::Matrix{Float64}, 
                                                  characteristics::TechnologySet)::Matrix{Float64}
    # Ensure dimensions match
    @assert size(weights_matrix, 2) == length(characteristics.A) "Weight and characteristic dimensions must align"
    
    # Combine A and eta into a characteristics matrix
    C = hcat(characteristics.A, characteristics.η)  # Size: n_vintages x 2
    
    # Matrix multiplication to get effective A and eta
    Effective = weights_matrix * C  # Size: n_combinations x 2
    
    return Effective  # Each row corresponds to a combination's (Aeff, eta_eff)
end

"""
Calculate cost based on effective characteristics
cost = A_eff - tau * eta_eff
"""
function compute_cost_matrix(effective_characteristics::Matrix{Float64}, tau::Float64)::Vector{Float64}
    return effective_characteristics[:,1] .- tau .* effective_characteristics[:,2]
end

"""
Enhanced 3D plotting function with density estimation
"""
function plot_portfolio_distributions_enhanced(sequence_matrix::Matrix{Float64})
    # Create figure with larger size and better resolution
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
        alpha=0.6,  # Add transparency
        legend=false,
        color=:blues,  # Use color gradient
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
    
    # Add summary statistics
    annotate!(p1, [(minimum(sequence_matrix[:,1]), 
                   minimum(sequence_matrix[:,2]), 
                   minimum(sequence_matrix[:,3]), 
                   "Period 0\nMean A_eff: $(round(mean(sequence_matrix[:,1]), digits=2))\n" *
                   "Mean η_eff: $(round(mean(sequence_matrix[:,2]), digits=2))")])

    annotate!(p2, [(minimum(sequence_matrix[:,4]), 
                   minimum(sequence_matrix[:,5]), 
                   minimum(sequence_matrix[:,6]), 
                   "Period 1\nMean A_eff: $(round(mean(sequence_matrix[:,4]), digits=2))\n" *
                   "Mean η_eff: $(round(mean(sequence_matrix[:,5]), digits=2))")])

    return plot_combined
end

"""
Convert portfolios to sequence matrix format
"""
function portfolios_to_matrix(portfolios::Vector{VintagePortfolio}, tau::Float64)::Matrix{Float64}
    n_portfolios = length(portfolios)
    sequence_matrix = Matrix{Float64}(undef, n_portfolios, 6)
    
    for (i, p) in enumerate(portfolios)
        # Get characteristics
        A_eff, η_eff = compute_effective_characteristics(p)
        cost = A_eff - tau * η_eff
        
        # Store in matrix
        sequence_matrix[i, 1] = A_eff
        sequence_matrix[i, 2] = η_eff
        sequence_matrix[i, 3] = cost
        
        # For now, use same values for period 1
        sequence_matrix[i, 4] = A_eff
        sequence_matrix[i, 5] = η_eff
        sequence_matrix[i, 6] = cost
    end
    
    return sequence_matrix
end

"""
Main execution function with Dirichlet sampling
"""
function run_example_dirichlet()
    # Define initial technologies with a smaller set for testing
    tech_minus1 = TechnologySet(
        A=[1.0, 0.7, 0.6],  # Just 2 technologies
        η=[1.0, 1.3, 1.2]
    )
    
    tech0 = TechnologySet(
        A=[1.4, 1.1, 1.3],
        η=[0.7, 0.9, 1.0]
    )
    
    tech1 = TechnologySet(
        A=[1.8, 1.5, 1.7],
        η=[0.4, 0.6, 0.8]
    )
    
    # Carbon price
    tau = 0.3
    
    println("Starting model solution with Dirichlet sampling...")
    
    # Generate portfolios for period 0
    portfolios0 = generate_vintage_portfolios_dirichlet(
        tech_minus1,
        tech0,
        n_samples=100,
        n_splits=20
    )

    # Generate two sets of period 1 portfolios to approximate the reachable domain
    portfolios1_p1 = generate_vintage_portfolios_dirichlet(
        tech0,
        tech1,
        n_samples=100,
        n_splits=20
    )

    portfolios1_p2 = generate_vintage_portfolios_dirichlet(
        tech_minus1,
        tech1, 
        n_samples=100, 
        n_splits=20
    )

    # Combine period 1 portfolios
    portfolios1 = vcat(portfolios1_p1, portfolios1_p2)
    
    println("Model solution complete. Processing results...")
    
    # Convert portfolios to matrix format
    matrix0 = portfolios_to_matrix(portfolios0, tau)
    matrix1 = portfolios_to_matrix(portfolios1, tau)
    
    # Replicate matrix0 to match dimensions with matrix1
    matrix0_expanded = repeat(matrix0, 2, 1)
    
    # Now combine matrices with matching dimensions
    sequence_matrix = hcat(matrix0_expanded[:, 1:3], matrix1[:, 1:3])
    
    # Print detailed statistics
    println("\nSequence Statistics:")
    println("Number of portfolios: ", size(sequence_matrix, 1))
    
    if size(sequence_matrix, 1) > 0
        println("A_eff_0 range: [$(round(minimum(sequence_matrix[:,1]), digits=3)), $(round(maximum(sequence_matrix[:,1]), digits=3))]")
        println("η_eff_0 range: [$(round(minimum(sequence_matrix[:,2]), digits=3)), $(round(maximum(sequence_matrix[:,2]), digits=3))]")
        println("Cost_0 range: [$(round(minimum(sequence_matrix[:,3]), digits=3)), $(round(maximum(sequence_matrix[:,3]), digits=3))]")
        println("A_eff_1 range: [$(round(minimum(sequence_matrix[:,4]), digits=3)), $(round(maximum(sequence_matrix[:,4]), digits=3))]")
        println("η_eff_1 range: [$(round(minimum(sequence_matrix[:,5]), digits=3)), $(round(maximum(sequence_matrix[:,5]), digits=3))]")
        println("Cost_1 range: [$(round(minimum(sequence_matrix[:,6]), digits=3)), $(round(maximum(sequence_matrix[:,6]), digits=3))]")
        
        # Generate and save plots
        p = plot_portfolio_distributions_enhanced(sequence_matrix)
        savefig(p, "portfolio_distributions_dirichlet.png")
        
        return (portfolios0, portfolios1, sequence_matrix, p)
    else
        println("No valid sequences generated")
        return (portfolios0, portfolios1, nothing, nothing)
    end
end

# Run example if file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_example_dirichlet()
end

"""
Find feasible period 1 portfolios given a period 0 portfolio
"""
function find_feasible_period1_domain(
    p0::VintagePortfolio,
    tech1::TechnologySet,
    tolerance::Float64=0.01
)::Vector{Float64}
    # Extract weights and characteristics from period 0
    w_existing = p0.existing_capital.weights
    w_new = p0.new_investment.weights
    
    # Calculate bounds for period 1
    # Lower bounds: Assuming complete retention of period 0 capital
    A_min = sum(w_existing .* p0.existing_capital.operation_mode.A) +
            sum(w_new .* minimum(tech1.A))
    η_max = sum(w_existing .* p0.existing_capital.operation_mode.η) +
            sum(w_new .* maximum(tech1.η))
    
    # Upper bounds: Assuming complete replacement with tech1
    A_max = maximum(tech1.A)
    η_min = minimum(tech1.η)
    
    return [A_min, A_max, η_min, η_max]
end

"""
Check if a period 1 portfolio is feasible given a period 0 portfolio
"""
function is_feasible_transition(
    p0::VintagePortfolio,
    p1::VintagePortfolio,
    tech1::TechnologySet,
    tolerance::Float64=0.01
)::Bool
    bounds = find_feasible_period1_domain(p0, tech1, tolerance)
    A_eff1, η_eff1 = compute_effective_characteristics(p1)
    
    return (
        A_eff1 >= bounds[1] - tolerance &&
        A_eff1 <= bounds[2] + tolerance &&
        η_eff1 >= bounds[3] - tolerance &&
        η_eff1 <= bounds[4] + tolerance
    )
end

"""
Find feasible period 0 portfolios that could lead to a given period 1 portfolio
"""
function find_feasible_period0_sources(
    p1::VintagePortfolio,
    portfolios0::Vector{VintagePortfolio},
    tech1::TechnologySet,
    tolerance::Float64=0.01
)::Vector{VintagePortfolio}
    feasible_sources = VintagePortfolio[]
    
    for p0 in portfolios0
        if is_feasible_transition(p0, p1, tech1, tolerance)
            push!(feasible_sources, p0)
        end
    end
    
    return feasible_sources
end

"""
Visualize feasible transitions between periods
"""
function plot_feasible_transitions(
    p0::VintagePortfolio,
    portfolios1::Vector{VintagePortfolio},
    tech1::TechnologySet,
    tau::Float64
)
    # Get characteristics for p0
    A_eff0, η_eff0 = compute_effective_characteristics(p0)
    cost0 = A_eff0 - tau * η_eff0
    
    # Find feasible domain bounds
    bounds = find_feasible_period1_domain(p0, tech1)
    
    # Separate feasible and infeasible period 1 portfolios
    feasible_p1 = VintagePortfolio[]
    infeasible_p1 = VintagePortfolio[]
    
    for p1 in portfolios1
        if is_feasible_transition(p0, p1, tech1)
            push!(feasible_p1, p1)
        else
            push!(infeasible_p1, p1)
        end
    end
    
    # Create plot
    p = scatter3d(
        title="Feasible Transitions from Period 0",
        xlabel="A_eff",
        ylabel="η_eff",
        zlabel="Cost"
    )
    
    # Plot period 0 point
    scatter3d!(p, [A_eff0], [η_eff0], [cost0],
        color=:blue,
        label="Period 0",
        markersize=6
    )
    
    # Plot feasible and infeasible period 1 points
    for portfolios in [feasible_p1, infeasible_p1]
        if !isempty(portfolios)
            characteristics = [compute_effective_characteristics(p) for p in portfolios]
            A_effs = [c[1] for c in characteristics]
            η_effs = [c[2] for c in characteristics]
            costs = [a - tau * e for (a, e) in characteristics]
            
            scatter3d!(p, A_effs, η_effs, costs,
                color=portfolios === feasible_p1 ? :green : :red,
                alpha=0.5,
                label=portfolios === feasible_p1 ? "Feasible P1" : "Infeasible P1"
            )
        end
    end
    
    return p
end
