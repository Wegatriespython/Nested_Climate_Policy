using Test
include("debug_portfolio.jl")

# Test for generate_weight_matrix_dirichlet
@testset "generate_weight_matrix_dirichlet" begin
    # Test basic properties for arbitrary dimensions
    n, n_samples = 5, 3
    result = generate_weight_matrix_dirichlet(n, n_samples; α=1.0)
    @test size(result) == (n_samples, n)
    @test all(abs.(sum(result, dims=2) .- 1.0) .< 1e-8)
    
    # Edge case tests
    @test_throws ErrorException generate_weight_matrix_dirichlet(0, 1; α=1.0)
end

# Test for generate_operation_modes
@testset "generate_operation_modes" begin
    # Test with controlled technology sets
    A_old = [0.5, 0.7]  # 2 old technologies
    η_old = [0.3, 0.4]
    A_new = [0.8, 0.9]  # 2 new technologies (both > old ones)
    η_new = [0.5, 0.6]
    
    old_tech = TechnologySet(A_old, η_old)
    new_tech = TechnologySet(A_new, η_new)

    result = generate_operation_modes(old_tech, new_tech)
    
    # Expected modes:
    # 1. Pure old tech modes: (0.5, 0.3), (0.7, 0.4)
    # 2. Combinations when new_tech.A >= old_tech.A with mode_factors [0.0, 1.0]
    #    For A_old[1]=0.5: with A_new[1]=0.8 and A_new[2]=0.9
    #    For A_old[2]=0.7: with A_new[1]=0.8 and A_new[2]=0.9
    
    # Test properties rather than exact values
    @test length(result.A) == length(result.η)  # Paired A and η values
    @test all(result.A .>= minimum(A_old))      # All A values >= min old tech
    @test all(result.A .<= maximum(A_new))      # All A values <= max new tech
    @test length(unique(result.A)) == length(result.A)  # All modes unique
end

# Test for compute_effective_characteristics
@testset "compute_effective_characteristics" begin
    # Generate random weights that sum to 1
    n_existing = 3
    n_new = 2
    weights_existing = rand(n_existing)
    weights_new = rand(n_new)
    total = sum(weights_existing) + sum(weights_new)
    weights_existing ./= total
    weights_new ./= total

    # Generate random technology characteristics
    A_existing = rand(n_existing)
    η_existing = rand(n_existing)
    A_new = rand(n_new)
    η_new = rand(n_new)

    operated_capital = OperatedCapital(
        weights=reshape(weights_existing, 1, :),
        base_tech=TechnologySet(A_existing, η_existing),
        operation_mode=TechnologySet(A_existing, η_existing)
    )
    new_investment = NewInvestment(
        weights=reshape(weights_new, 1, :),
        technologies=TechnologySet(A_new, η_new)
    )
    vintage_portfolio = VintagePortfolio(existing_capital=operated_capital, new_investment=new_investment)

    A_eff_expected = dot(weights_existing, A_existing) + dot(weights_new, A_new)
    η_eff_expected = dot(weights_existing, η_existing) + dot(weights_new, η_new)

    A_eff, η_eff = compute_effective_characteristics(vintage_portfolio)

    @test isapprox(A_eff, A_eff_expected, atol=1e-6)
    @test isapprox(η_eff, η_eff_expected, atol=1e-6)
end

# Test for generate_vintage_portfolios_dirichlet
@testset "generate_vintage_portfolios_dirichlet" begin
    n_tech = 3  # Number of technologies
    tech_minus1 = TechnologySet(rand(n_tech), rand(n_tech))
    tech0 = TechnologySet(rand(n_tech), rand(n_tech))

    n_samples, n_splits = 4, 3
    portfolios = generate_vintage_portfolios_dirichlet(tech_minus1, tech0, n_samples=n_samples, n_splits=n_splits)

    # The actual number of portfolios will be n_samples * (n_splits - 2)
    # because the function skips splits ≈ 0.0 and ≈ 1.0
    expected_portfolio_count = n_samples * (n_splits - 2)
    @test length(portfolios) == expected_portfolio_count

    for portfolio in portfolios
        # The operation modes determine the actual number of technologies
        n_operated = length(portfolio.existing_capital.operation_mode.A)
        n_new = length(portfolio.new_investment.technologies.A)
        
        @test size(portfolio.existing_capital.weights) == (1, n_operated)
        @test size(portfolio.new_investment.weights) == (1, n_new)
        @test isapprox(sum(portfolio.existing_capital.weights) + sum(portfolio.new_investment.weights), 1.0, atol=1e-6)
    end
end

# Test for portfolios_to_matrix
@testset "portfolios_to_matrix" begin
    n_tech = 3
    tech_minus1 = TechnologySet(rand(n_tech), rand(n_tech))
    tech0 = TechnologySet(rand(n_tech), rand(n_tech))
    
    # Create sample portfolios using random weights
    n_portfolios = 4
    portfolios = Vector{VintagePortfolio}()  # Explicitly specify type
    for _ in 1:n_portfolios
        w_existing = rand(n_tech)
        w_new = rand(n_tech)
        total = sum(w_existing) + sum(w_new)
        w_existing ./= total
        w_new ./= total
        
        portfolio = VintagePortfolio(
            existing_capital=OperatedCapital(
                weights=reshape(w_existing, 1, :),
                base_tech=tech_minus1,
                operation_mode=tech_minus1
            ),
            new_investment=NewInvestment(
                weights=reshape(w_new, 1, :),
                technologies=tech0
            )
        )
        push!(portfolios, portfolio)
    end

    tau = 0.3
    matrix = portfolios_to_matrix(portfolios, tau)

    @test size(matrix) == (n_portfolios, 3)
    for i in 1:n_portfolios
        A_eff, η_eff = compute_effective_characteristics(portfolios[i])
        @test isapprox(matrix[i, 1], A_eff, atol=1e-6)
        @test isapprox(matrix[i, 2], η_eff, atol=1e-6)
        @test isapprox(matrix[i, 3], A_eff - tau * η_eff, atol=1e-6)
    end
end

# Test for run_example
@testset "run_example" begin
    # Run the example and check the outputs
    portfolios0, portfolios1, sequence_matrix, p = run_example()

    @test length(portfolios0) > 0
    @test length(portfolios1) > 0
    @test size(sequence_matrix, 1) > 0
end