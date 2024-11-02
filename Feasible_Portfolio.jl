using Parameters
using Distributions  # For Dirichlet sampling
using Plots  # For visualization
using LinearAlgebra       

@with_kw struct TechnologySet
    A::Vector{Float64}
    η::Vector{Float64}
end

@with_kw struct OperatedCapital
    weights::Matrix{Float64}
    base_tech::TechnologySet
    operation_mode::TechnologySet
end

@with_kw struct NewInvestment
    weights::Matrix{Float64}
    technologies::TechnologySet
end

@with_kw struct VintagePortfolio
    existing_capital::OperatedCapital
    new_investment::NewInvestment
end

function generate_weight_matrix_dirichlet(
    n::Int, 
    n_samples::Int=100; 
    α::Float64=1.0
)::Matrix{Float64}
    if n < 1
        error("Number of technologies must be at least 1.")
    end
    if n == 1
        return reshape([1.0], 1, 1)
    end
    
    dist = Dirichlet(n, α)
    samples = rand(dist, n_samples)'
    return samples
end

function generate_operation_modes(old_tech::TechnologySet, new_tech::TechnologySet)::TechnologySet
    mode_factors = [0.0, 1.0]
    
    # Use Set of tuples to keep A and η paired
    modes = Set{Tuple{Float64, Float64}}()
    
    println("\nGenerating operation modes:")
    # For each old technology
    for i in 1:length(old_tech.A)
        # Add pure old tech mode
        push!(modes, (old_tech.A[i], old_tech.η[i]))
        
        for j in 1:length(new_tech.A)
            if new_tech.A[j] >= old_tech.A[i]
                for mode in mode_factors
                    A_mode = old_tech.A[i] * (1 - mode) + new_tech.A[j] * mode
                    η_mode = old_tech.η[i] * (1 - mode) + new_tech.η[j] * mode
                    push!(modes, (A_mode, η_mode))
                end
            end
        end
    end
    
    # Unzip the tuples into separate vectors
    A_composite = Float64[]
    η_composite = Float64[]
    for (a, η) in modes
        push!(A_composite, a)
        push!(η_composite, η)
    end
    
   
    return TechnologySet(A_composite, η_composite)
end

function compute_effective_characteristics(p::VintagePortfolio)::Tuple{Float64, Float64}
    println("\nComputing effective characteristics:")
    
  
    
    total_existing = sum(p.existing_capital.weights)
    total_new = sum(p.new_investment.weights)
    total = total_existing + total_new
    
    

    # Convert weights to vectors ensuring correct dimensions
    weights_vec = vec(p.existing_capital.weights)
    @assert length(weights_vec) == length(p.existing_capital.operation_mode.A) "Weight and value dimensions don't match"
    
    weighted_A = weights_vec .* p.existing_capital.operation_mode.A
    running_sum = 0.0
    for i in 1:length(weighted_A)
        running_sum += weighted_A[i]
    end
    
    A_existing = running_sum / total_existing
    
    # Simi  lar detailed debugging for A_new

    new_weights_vec = vec(p.new_investment.weights)
    weighted_A_new = new_weights_vec .* p.new_investment.technologies.A

    running_sum_new = sum(weighted_A_new)
    
    A_new = running_sum_new / total_new

    A_eff = A_existing * total_existing + A_new * total_new

    
    # Similar calculations for η
    η_existing = sum(vec(p.existing_capital.weights) .* p.existing_capital.operation_mode.η) / total_existing
    η_new = sum(vec(p.new_investment.weights) .* p.new_investment.technologies.η) / total_new
    η_eff = η_existing * total_existing + η_new * total_new
    
    return (A_eff, η_eff)
end

function generate_vintage_portfolios_dirichlet(
    tech_minus1::TechnologySet,
    tech0::TechnologySet;
    n_samples::Int=50,
    n_splits::Int=10
)::Vector{VintagePortfolio}
    portfolios = Vector{VintagePortfolio}()
    operated_tech = generate_operation_modes(tech_minus1, tech0)
    
    n_operated = length(operated_tech.A)
    n_new = length(tech0.A)
    
    println("Debug: Generating samples for $n_operated operated modes and $n_new new technologies")
    
    splits = LinRange(0, 1, n_splits)
    
    for split in splits
        if split ≈ 0.0 || split ≈ 1.0
            continue
        end
        
        # Generate n_samples × n matrices
        operated_weights = generate_weight_matrix_dirichlet(n_operated, n_samples)  # n_samples × n_operated
        new_weights = generate_weight_matrix_dirichlet(n_new, n_samples)           # n_samples × n_new
        
        println("Debug dimensions:")
        println("operated_weights: $(size(operated_weights))")
        println("new_weights: $(size(new_weights))")
        
        for i in 1:n_samples
            # Scale weights maintaining matrix dimensions
            op_weights_scaled = operated_weights[i,:] .* (split / sum(operated_weights[i,:]))
            new_weights_scaled = new_weights[i,:] .* ((1-split) / sum(new_weights[i,:]))
            
            # Create portfolio with 1 × n matrices for weights
            portfolio = VintagePortfolio(
                existing_capital=OperatedCapital(
                    weights=reshape(op_weights_scaled, 1, :),  # 1 × n_operated
                    base_tech=tech_minus1,
                    operation_mode=operated_tech
                ),
                new_investment=NewInvestment(
                    weights=reshape(new_weights_scaled, 1, :),  # 1 × n_new
                    technologies=tech0
                )
            )
            
            # Verify dimensions
            @assert size(portfolio.existing_capital.weights, 2) == n_operated "Operated weights dimension mismatch"
            @assert size(portfolio.new_investment.weights, 2) == n_new "New weights dimension mismatch"
            
            push!(portfolios, portfolio)
        end
    end
    
    return portfolios
end



function portfolios_to_matrix(portfolios::Vector{VintagePortfolio}, tau::Float64)::Matrix{Float64}
    n_portfolios = length(portfolios)
    sequence_matrix = Matrix{Float64}(undef, n_portfolios, 3)
    
    for (i, p) in enumerate(portfolios)
        A_eff, η_eff = compute_effective_characteristics(p)
        cost = A_eff - tau * η_eff
        
        sequence_matrix[i, 1] = A_eff
        sequence_matrix[i, 2] = η_eff
        sequence_matrix[i, 3] = cost
    end
    
    return sequence_matrix
end




# Modify run_example() to use the new mapping approach
function run_example()
    # Define technologies
    tech_minus1 = TechnologySet(
        A=[1.0, 0.7, 0.6],
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
    
    tau = 0.3
    
    println("Starting model solution...")
    
    # Generate portfolios
    portfolios0 = generate_vintage_portfolios_dirichlet(tech_minus1, tech0)
    portfolios1 = vcat(
        generate_vintage_portfolios_dirichlet(tech0, tech1),
        generate_vintage_portfolios_dirichlet(tech_minus1, tech1)
    )
    
    # Convert to sequence matrix
    matrix0 = portfolios_to_matrix(portfolios0, tau)
    matrix1 = portfolios_to_matrix(portfolios1, tau)
    matrix0_expanded = repeat(matrix0, 2, 1)
    sequence_matrix = hcat(matrix0_expanded, matrix1)
    
    # Generate moment matrices
    M_Hole_0 = generate_hole_moments_matrix(sequence_matrix, 0)
    M_Hole_1 = generate_hole_moments_matrix(sequence_matrix, 1)
    
    M_Fixed_0 = generate_fixed_moments_matrix(sequence_matrix, 0)
    M_Fixed_1 = generate_fixed_moments_matrix(sequence_matrix, 1)
    
    # Export as vectors
    moments_vec_0 = export_moments_vector(M_Fixed_0)
    moments_vec_1 = export_moments_vector(M_Fixed_1)
    
    # Test inversion
    test_point = M_Fixed_0[50, :]  # Take a point from the middle
    recovered_portfolio = invert_moments_to_portfolio(
        test_point[1], 
        test_point[2], 
        portfolios0
    )
    
    return (
        M_Hole_0, M_Hole_1,
        M_Fixed_0, M_Fixed_1,
        moments_vec_0, moments_vec_1,
        recovered_portfolio
    )
end

# Run if file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_example()
end 