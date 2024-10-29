using Test
include("model_core_partial_investment.jl")
using .ModelParametersModule: ModelParameters

function debug_investment_model()
    # Base parameters for testing
    test_params = ModelParameters(
        β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
        δ = 0.1, γ = 0.01, K_init = 1.0,
        μ_A = 1.0, μ_η = 1.0, σ_A = 0.2, σ_η = 0.2, ρ = 0.5,
        θ_min = 0.1, θ_max = 1.0
    )

    # Test Case 1: Baseline with small differences
    println("\nTest 1: Baseline - Small Technology Difference")
    try
        A_0, η_0 = 1.0, 1.0
        A_1, η_1 = 1.01, 1.001  # Very small differences
        skill_factor = 1.0

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")

        result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)
        
        println("\nResults:")
        println("Production:")
        println("  Y_0: $(round(result["Y_0"], digits=4))")
        println("  Y_1: $(round(result["Y_1"], digits=4))")
        println("\nTechnology:")
        println("  Split: $(round(result["Technology_Split"] * 100, digits=2))%")
        println("  A_1 effective: $(round(result["A_1_eff"], digits=4))")
        println("  η_1 effective: $(round(result["η_1_eff"], digits=4))")
        println("\nLabor:")
        println("  Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        println("  Δη: $(round(result["Δη"], digits=4))")
    catch e
        println("Error: ", e)
    end

    # Test Case 2: Distinct technologies
    println("\nTest 2: Distinct Technologies")
    try
        A_0, η_0 = 1.0, 1.0
        A_1, η_1 = 1.5, 0.8  # More significant differences
        skill_factor = 1.0

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")

        result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)
        
        println("\nResults:")
        println("Production:")
        println("  Y_0: $(round(result["Y_0"], digits=4))")
        println("  Y_1: $(round(result["Y_1"], digits=4))")
        println("\nTechnology:")
        println("  Split: $(round(result["Technology_Split"] * 100, digits=2))%")
        println("  A_1 effective: $(round(result["A_1_eff"], digits=4))")
        println("  η_1 effective: $(round(result["η_1_eff"], digits=4))")
        println("\nLabor:")
        println("  Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        println("  Δη: $(round(result["Δη"], digits=4))")
    catch e
        println("Error: ", e)
    end

    # Test Case 3: With lower initial technology
    println("\nTest 3: Lower Initial Technology")
    try
        A_0, η_0 = 0.8, 0.7
        A_1, η_1 = 1.0, 1.0
        skill_factor = 0.5

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")

        result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)
        
        println("\nResults:")
        println("Production:")
        println("  Y_0: $(round(result["Y_0"], digits=4))")
        println("  Y_1: $(round(result["Y_1"], digits=4))")
        println("\nTechnology:")
        println("  Split: $(round(result["Technology_Split"] * 100, digits=2))%")
        println("  A_1 effective: $(round(result["A_1_eff"], digits=4))")
        println("  η_1 effective: $(round(result["η_1_eff"], digits=4))")
        println("\nLabor:")
        println("  Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        println("  Δη: $(round(result["Δη"], digits=4))")
    catch e
        println("Error: ", e)
    end

    # Test Case 4: With tax
    println("\nTest 4: With Carbon Tax")
    try
        A_0, η_0 = 1.0, 1.0
        A_1, η_1 = 1.2, 0.8
        skill_factor = 1.0
        tax_rate = 0.1

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")
        println("tax_rate: $tax_rate")

        result = compute_equilibrium_core(tax_rate, A_0, η_0, A_1, η_1, skill_factor, test_params)
        
        println("\nResults:")
        println("Production:")
        println("  Y_0: $(round(result["Y_0"], digits=4))")
        println("  Y_1: $(round(result["Y_1"], digits=4))")
        println("\nTechnology:")
        println("  Split: $(round(result["Technology_Split"] * 100, digits=2))%")
        println("  A_1 effective: $(round(result["A_1_eff"], digits=4))")
        println("  η_1 effective: $(round(result["η_1_eff"], digits=4))")
        println("\nLabor:")
        println("  Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        println("  Δη: $(round(result["Δη"], digits=4))")
    catch e
        println("Error: ", e)
    end

    # Test Case 5: Technology transition test
    println("\nTest 5: Technology Transition")
    try
        A_0, η_0 = 1.0, 1.0
        A_1, η_1 = 1.1, 0.9
        skill_factor = 1.0

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")

        result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)
        
        println("\nResults:")
        println("Production:")
        println("  Y_0: $(round(result["Y_0"], digits=4))")
        println("  Y_1: $(round(result["Y_1"], digits=4))")
        println("\nTechnology:")
        println("  Split: $(round(result["Technology_Split"] * 100, digits=2))%")
        println("  A_1 effective: $(round(result["A_1_eff"], digits=4))")
        println("  η_1 effective: $(round(result["η_1_eff"], digits=4))")
        println("\nLabor:")
        println("  Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        println("  Δη: $(round(result["Δη"], digits=4))")
    catch e
        println("Error: ", e)
    end
end

# Run the debug tests
debug_investment_model()