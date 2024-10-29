using Test
include("model_core_skill_fix.jl")  # Include the model file
using .ModelParametersModule: ModelParameters, DEFAULT_PARAMS
import .ModelParametersModule: PolicyExpectations, form_tax_expectations

# Test function to debug the equilibrium computation
function debug_equilibrium_computation()
    # 1. First try with original parameters and zero tax
    println("Test 1: Basic case with zero tax")
    try
        expectations = PolicyExpectations(
            0.0,    # current_tax
            0.0,    # announced_tax
            1.0,    # η_mean
            0.2,    # η_std
            1.0     # credibility
        )
        result = compute_equilibrium(expectations, DEFAULT_PARAMS)
        if !isempty(result)
            println("✓ Basic case succeeded")
            println("Sample values:")
            println("  C_0: $(result["C_0"])")
            println("  Y_0: $(result["Y_t"])")
            println("  L_0: $(result["L_t"])")
            println("  η_0: $(result["η_t"])")
        else
            println("✗ Basic case failed: Empty result")
        end
    catch e
        println("✗ Basic case failed: $e")
        println("Error details: ")
        Base.showerror(stdout, e, catch_backtrace())
    end

    # 2. Test with increasing tax rates
    println("\nTest 2: Tax rate sensitivity")
    test_params = ModelParameters(
        β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
        δ = 0.1, γ = 0.01, K_init = 1.0, μ_A = 1.0,  # Reduced γ for stability
        μ_η = 1.0, σ_A = 0.2, σ_η = 0.2, ρ = 0.5,
        θ_min = 0.1, θ_max = 1.0
    )
    
    tax_rates = [0.0, 0.02, 0.04, 0.06, 0.08]
    for τ in tax_rates
        println("\nTesting with tax rate τ = $τ")
        try
            expectations = PolicyExpectations(τ, τ, 1.0, 0.2, 1.0)
            result = compute_equilibrium(expectations, test_params)
            if !isempty(result)
                println("✓ Succeeded with τ = $τ")
                println("Key metrics:")
                println("  Y_0: $(result["Y_t"])")
                println("  C_0: $(result["C_0"])")
                println("  L_0: $(result["L_t"])")
                println("  η_0: $(result["η_t"])")
            else
                println("✗ Failed with τ = $τ: Empty result")
            end
        catch e
            println("✗ Failed with τ = $τ: $e")
        end
    end

    # 3. Labor Efficiency Tests
    println("\nTest 3: Labor Efficiency Channel")
    try
        # Fixed technology values for consistency
        A_0 = 1.0
        η_0 = 1.0
        
        # Test different technology changes
        Δη_values = [0.0, 0.1, 0.2, 0.3]
        skill_factors = [0.5, 1.0, 2.0]
        
        for sf in skill_factors
            println("\nTesting with skill_factor = $sf")
            for Δη in Δη_values
                η_1 = η_0 + Δη
                println("\n  Technology change Δη = $Δη")
                try
                    result = compute_equilibrium_core(0.0, A_0, η_0, A_0, η_1, sf, test_params)
                    if !isempty(result)
                        println("  ✓ Succeeded")
                        println("  Key metrics:")
                        println("    Y_0: $(result["Y_0"])")
                        println("    L_0: $(result["L_0"])")
                        println("    w_0: $(result["w_0"])")
                        println("    Labor_Efficiency: $(result["Labor_Efficiency"])")
                    else
                        println("  ✗ Failed: Empty result")
                    end
                catch e
                    println("  ✗ Failed with Δη = $Δη")
                    println("  Error: $e")
                end
            end
        end
        
    catch e
        println("✗ Labor efficiency test failed to run")
        println("Error details: ")
        Base.showerror(stdout, e, catch_backtrace())
    end

    # 4. Test with computed skill factor
    println("\nTest 4: Computed Skill Factor")
    try
        computed_sf = compute_aggregate_skill(test_params)
        println("Computed skill_factor = $computed_sf")
        
        # Test with moderate technology change
        A_0 = 1.0
        η_0 = 1.0
        η_1 = 1.2  # 20% increase in carbon intensity
        
        result = compute_equilibrium_core(0.0, A_0, η_0, A_0, η_1, computed_sf, test_params)
        if !isempty(result)
            println("✓ Succeeded with computed skill factor")
            println("Key metrics:")
            println("  Y_0: $(result["Y_0"])")
            println("  L_0: $(result["L_0"])")
            println("  w_0: $(result["w_0"])")
            println("  Labor_Efficiency: $(result["Labor_Efficiency"])")
        else
            println("✗ Failed: Empty result")
        end
    catch e
        println("✗ Computed skill factor test failed")
        println("Error details: ")
        Base.showerror(stdout, e, catch_backtrace())
    end

    # 5. MCTS Failure Case Reproduction
    println("\nTest 5: MCTS Failure Case Reproduction")
    try
        # Use the basic parameters without MCTS-specific ones
        test_params = ModelParameters(
            β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
            δ = 0.1, γ = 0.01, K_init = 1.0,
            μ_A = 1.0, μ_η = 1.0, σ_A = 0.2, σ_η = 0.2, ρ = 0.5,
            θ_min = 0.1, θ_max = 1.0
        )

        # Test case 1: First failure case
        println("\nTesting first failure case:")
        A_0 = 0.72713302959613
        η_0 = 0.7424202688987085
        A_1 = 0.7544360907541691
        η_1 = 0.5710404277959683
        skill_factor = 2.5584278811044947

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")
        println("γ: $(test_params.γ)")
        println("Expected labor efficiency: $(1.0 / (1.0 + test_params.γ * ((η_1 - η_0)^2) * skill_factor))")

        result1 = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)

        # Test case 2: Second failure case
        println("\nTesting second failure case:")
        A_0 = 0.7896398370594069
        η_0 = 0.9103110796535978
        A_1 = 0.8031267294454324
        η_1 = 0.8457758930867435
        skill_factor = 2.5584278811044947

        println("Parameters:")
        println("A_0: $A_0, η_0: $η_0")
        println("A_1: $A_1, η_1: $η_1")
        println("skill_factor: $skill_factor")
        println("γ: $(test_params.γ)")
        println("Expected labor efficiency: $(1.0 / (1.0 + test_params.γ * ((η_1 - η_0)^2) * skill_factor))")

        result2 = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, skill_factor, test_params)

        # Print results if successful
        if !isempty(result1) && !isempty(result2)
            println("\nBoth cases succeeded!")
            println("\nFirst case results:")
            println("  Y_0: $(result1["Y_0"])")
            println("  L_0: $(result1["L_0"])")
            println("  Labor_Efficiency: $(result1["Labor_Efficiency"])")
            
            println("\nSecond case results:")
            println("  Y_0: $(result2["Y_0"])")
            println("  L_0: $(result2["L_0"])")
            println("  Labor_Efficiency: $(result2["Labor_Efficiency"])")
        end

    catch e
        println("✗ MCTS failure case reproduction failed")
        println("Error details: ")
        Base.showerror(stdout, e, catch_backtrace())
    end

    # 5b. MCTS Failure Case Diagnosis
    println("\nTest 5b: MCTS Failure Case Diagnosis")
    try
        test_params = ModelParameters(
            β = 0.96, σ = 2.0, χ = 1.0, ν = 1.0, α = 0.33,
            δ = 0.1, γ = 0.01, K_init = 1.0,
            μ_A = 1.0, μ_η = 1.0, σ_A = 0.2, σ_η = 0.2, ρ = 0.5,
            θ_min = 0.1, θ_max = 1.0
        )

        # Test case variations
        A_0 = 0.72713302959613
        η_0 = 0.7424202688987085
        A_1 = 0.7544360907541691
        η_1 = 0.5710404277959683
        skill_factor = 2.5584278811044947

        # Test 1: Original A values but smaller Δη
        println("\nTest 1: Original A values, smaller Δη")
        η_1_small = η_0 - 0.05  # Smaller technology change
        try
            result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1_small, skill_factor, test_params)
            println("✓ Succeeded with smaller Δη")
            println("Labor_Efficiency: $(result["Labor_Efficiency"])")
        catch e
            println("✗ Failed with smaller Δη")
        end

        # Test 2: Higher A values but original Δη
        println("\nTest 2: Higher A values, original Δη")
        A_0_high = 1.0
        A_1_high = A_1 * (1.0/A_0)  # Keep same relative change
        try
            result = compute_equilibrium_core(0.0, A_0_high, η_0, A_1_high, η_1, skill_factor, test_params)
            println("✓ Succeeded with higher A")
            println("Labor_Efficiency: $(result["Labor_Efficiency"])")
        catch e
            println("✗ Failed with higher A")
        end

        # Test 3: Original case with lower skill factor
        println("\nTest 3: Original case, lower skill factor")
        try
            result = compute_equilibrium_core(0.0, A_0, η_0, A_1, η_1, 1.0, test_params)
            println("✓ Succeeded with lower skill factor")
            println("Labor_Efficiency: $(result["Labor_Efficiency"])")
        catch e
            println("✗ Failed with lower skill factor")
        end

        # Test 4: Original case with adjusted bounds
        println("\nTest 4: Original case, wider bounds")
        potential_Y0 = (1.0 - 0.0 * η_0) * A_0  # Base case
        println("Potential Y0: $potential_Y0")
        println("Labor efficiency: $(1.0 / (1.0 + test_params.γ * ((η_1 - η_0)^2) * skill_factor))")

    catch e
        println("✗ Diagnosis tests failed to run")
        println("Error details: ")
        Base.showerror(stdout, e, catch_backtrace())
    end
end

# Run the debug tests
debug_equilibrium_computation()