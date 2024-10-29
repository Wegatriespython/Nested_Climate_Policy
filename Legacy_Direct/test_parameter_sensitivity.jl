using Test
include("model_core_partial_investment.jl")
using .ModelParametersModule: ModelParameters

function test_parameter_sensitivity()
    # Base case parameters (from the failing case)
    base_A_0 = 1.4766509578753597
    base_η_0 = 1.639805004396013
    base_skill_factor = 2.5584278811044947
    base_tax = 0.1

    # Test 1: Skill Factor Sensitivity
    println("\nTest 1: Skill Factor Sensitivity")
    skill_factors = [0.5, 1.0, 1.5, 2.0, 2.5584278811044947]
    for sf in skill_factors
        try
            println("\nTrying skill_factor: $sf")
            result = compute_equilibrium_core(
                base_tax, base_A_0, base_η_0, base_A_0, base_η_0,
                sf, DEFAULT_PARAMS
            )
            println("Success with skill_factor = $sf")
            println("Labor Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        catch e
            println("Failed with skill_factor = $sf")
            println("Error: ", typeof(e))
        end
    end

    # Test 2: Carbon Intensity Sensitivity
    println("\nTest 2: Carbon Intensity (η) Sensitivity")
    η_values = [0.5, 1.0, 1.5, base_η_0]
    for η in η_values
        try
            println("\nTrying η: $η")
            result = compute_equilibrium_core(
                base_tax, base_A_0, η, base_A_0, η,
                1.0, DEFAULT_PARAMS
            )
            println("Success with η = $η")
            println("Effective Output: $(round(result["Y_0"], digits=4))")
        catch e
            println("Failed with η = $η")
            println("Error: ", typeof(e))
        end
    end

    # Test 3: Tax Rate Sensitivity
    println("\nTest 3: Tax Rate Sensitivity")
    tax_rates = [0.0, 0.05, 0.1, 0.15, 0.2]
    for τ in tax_rates
        try
            println("\nTrying tax rate: $τ")
            result = compute_equilibrium_core(
                τ, base_A_0, base_η_0, base_A_0, base_η_0,
                1.0, DEFAULT_PARAMS
            )
            println("Success with τ = $τ")
            println("Output: $(round(result["Y_0"], digits=4))")
        catch e
            println("Failed with τ = $τ")
            println("Error: ", typeof(e))
        end
    end

    # Test 4: Productivity Sensitivity
    println("\nTest 4: Productivity (A) Sensitivity")
    A_values = [0.5, 1.0, base_A_0, 2.0]
    for A in A_values
        try
            println("\nTrying A: $A")
            result = compute_equilibrium_core(
                base_tax, A, base_η_0, A, base_η_0,
                1.0, DEFAULT_PARAMS
            )
            println("Success with A = $A")
            println("Output: $(round(result["Y_0"], digits=4))")
        catch e
            println("Failed with A = $A")
            println("Error: ", typeof(e))
        end
    end

    # Test 5: Combined Parameter Sensitivity
    println("\nTest 5: Testing Parameter Combinations")
    test_cases = [
        (A=1.0, η=1.0, sf=1.0, τ=0.1),  # Base case
        (A=base_A_0, η=1.0, sf=base_skill_factor, τ=0.1),  # High A, normal η
        (A=1.0, η=base_η_0, sf=1.0, τ=0.1),  # Normal A, high η
        (A=base_A_0, η=base_η_0, sf=1.0, τ=0.1),  # High A, high η
    ]
    
    for case in test_cases
        try
            println("\nTrying combination:")
            println("A: $(case.A), η: $(case.η), skill_factor: $(case.sf), τ: $(case.τ)")
            result = compute_equilibrium_core(
                case.τ, case.A, case.η, case.A, case.η,
                case.sf, DEFAULT_PARAMS
            )
            println("Success!")
            println("Output: $(round(result["Y_0"], digits=4))")
            println("Labor Efficiency: $(round(result["Labor_Efficiency"], digits=4))")
        catch e
            println("Failed!")
            println("Error: ", typeof(e))
        end
    end
end

# Run the tests
test_parameter_sensitivity() 