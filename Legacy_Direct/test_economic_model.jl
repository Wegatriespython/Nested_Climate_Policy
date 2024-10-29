include("model_core.jl")

# Add the missing tax expectations function
function form_tax_expectations(current_tax::Float64, announced_tax::Float64, 
                             η_mean::Float64, η_std::Float64, credibility::Float64)
    # Simple weighted average based on credibility
    expected_tax = current_tax * (1 - credibility) + announced_tax * credibility
    return expected_tax
end

function test_economic_model()
    try
        params = DEFAULT_PARAMS
        τ = 0.2  # Example carbon tax rate
        result = compute_equilibrium(τ, params)
        
        println("\nEquilibrium Results:")
        
        println("\nAdjustment Parameters:")
        println("  Aggregate Skill Factor: $(round(result["Skill_Factor"], digits=4))")
        println("  Adjustment Cost: $(round(result["Adj_Cost"], digits=4))")
        
        println("\nTechnology Parameters:")
        println("  Period 0 Productivity (A_0): $(round(result["A_0"], digits=4))")
        println("  Period 0 Carbon Intensity (η_0): $(round(result["η_0"], digits=4))")
        println("  Period 1 Productivity (A_1): $(round(result["A_1"], digits=4))")
        println("  Period 1 Carbon Intensity (η_1): $(round(result["η_1"], digits=4))")
        
        println("\nPeriod 0 (Current):")
        println("  Output (Y_0): $(round(result["Y_0"], digits=4))")
        println("  Consumption (C_0): $(round(result["C_0"], digits=4))")
        println("  Capital (K_0): $(round(result["K_0"], digits=4))")
        println("  Labor (L_0): $(round(result["L_0"], digits=4))")
        println("  Wage (w_0): $(round(result["w_0"], digits=4))")
        println("  Carbon Tax Revenue: $(round(result["Tax_0"], digits=4))")
        
        println("\nPeriod 1 (Future):")
        println("  Output (Y_1): $(round(result["Y_1"], digits=4))")
        println("  Consumption (C_1): $(round(result["C_1"], digits=4))")
        println("  Capital (K_1): $(round(result["K_1"], digits=4))")
        println("  Labor (L_1): $(round(result["L_1"], digits=4))")
        println("  Wage (w_1): $(round(result["w_1"], digits=4))")
        println("  Interest Rate (r_1): $(round(result["r_1"], digits=4))")
        println("  Carbon Tax Revenue: $(round(result["Tax_1"], digits=4))")
        
        println("\nGrowth Rates:")
        println("  Capital Growth: $(round(result["K_1"]/result["K_0"] - 1, digits=4))")
        println("  Consumption Growth: $(round(result["C_1"]/result["C_0"] - 1, digits=4))")
        println("  Productivity Growth: $(round(result["A_1"]/result["A_0"] - 1, digits=4))")
        println("  Carbon Intensity Change: $(round(result["η_1"]/result["η_0"] - 1, digits=4))")
    catch e
        println("Error computing equilibrium:")
        println(e)
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_economic_model()
end
