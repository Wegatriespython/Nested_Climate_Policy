using Plots
using ForwardDiff
using Statistics
using LinearAlgebra
using Measures

"""
Represents a discrete technology set with productivity (A) and emissions (η) values
"""
struct TechnologySet
    A::Vector{Float64}  # Productivity values
    η::Vector{Float64}  # Emissions intensity values
end

"""
Safe sigmoid function to prevent numerical overflow
"""
function safe_sigmoid(x::T) where T<:Real
    if x > 20
        return one(T)
    elseif x < -20
        return zero(T)
    else
        return 1 / (1 + exp(-x))
    end
end

"""
Bijective mapping from normalized space [0,1] to technology space
Uses piecewise linear interpolation with smooth transitions
"""
function map_to_tech_space(x::T, tech_points::Vector{Float64}; σ::Float64=0.1) where T<:Real
    # Input validation
    if !(0 ≤ x ≤ 1)
        return convert(T, NaN)
    end
    
    sorted_points = sort(tech_points)
    n_points = length(sorted_points)
    
    # Handle edge cases
    if x ≈ 0
        return convert(T, sorted_points[1])
    elseif x ≈ 1
        return convert(T, sorted_points[end])
    end
    
    # Linear interpolation with smooth transitions
    segment_size = 1 / (n_points - 1)
    segment_idx = floor(Int, x * (n_points - 1)) + 1
    segment_idx = min(segment_idx, n_points - 1)
    
    local_x = (x - (segment_idx-1)*segment_size) / segment_size
    
    # Smooth transition using modified sigmoid
    α = safe_sigmoid(10 * (local_x - 0.5))
    
    return (1 - α) * sorted_points[segment_idx] + α * sorted_points[segment_idx + 1]
end

"""
Inverse mapping from technology space back to normalized space [0,1]
"""
function map_from_tech_space(y::T, tech_points::Vector{Float64}; σ::Float64=0.1) where T<:Real
    sorted_points = sort(tech_points)
    
    # Input validation
    if !(sorted_points[1] ≤ y ≤ sorted_points[end])
        return convert(T, NaN)
    end
    
    # Handle edge cases
    if y ≈ sorted_points[1]
        return zero(T)
    elseif y ≈ sorted_points[end]
        return one(T)
    end
    
    # Find appropriate segment
    segment_idx = 1
    for i in 2:length(sorted_points)
        if y ≤ sorted_points[i]
            segment_idx = i-1
            break
        end
    end
    
    # Linear interpolation within segment
    segment_size = 1 / (length(sorted_points) - 1)
    local_y = (y - sorted_points[segment_idx]) / (sorted_points[segment_idx + 1] - sorted_points[segment_idx])
    
    # Smooth transition using modified sigmoid
    α = safe_sigmoid(10 * (local_y - 0.5))
    
    return (segment_idx - 1) * segment_size + α * segment_size
end

"""
Test the bijectivity of the mapping functions
"""
function test_bijectivity(tech_points::Vector{Float64}; n_test_points::Int=100)
    test_points = range(0, 1, length=n_test_points)
    max_error = 0.0
    errors = Float64[]
    
    for x in test_points
        # Forward mapping
        y = map_to_tech_space(x, tech_points)
        # Inverse mapping
        x_recovered = map_from_tech_space(y, tech_points)
        # Calculate error
        error = abs(x - x_recovered)
        push!(errors, error)
        max_error = max(max_error, error)
    end
    
    return max_error, mean(errors), std(errors)
end

"""
Compute gradients of the mapping functions using ForwardDiff
"""
function compute_gradients(tech_points::Vector{Float64}; n_test_points::Int=100)
    test_points = range(0.01, 0.99, length=n_test_points)  # Avoid edge points
    
    # Forward mapping gradients
    forward_grads = [ForwardDiff.derivative(x -> map_to_tech_space(x, tech_points), x) for x in test_points]
    
    # Inverse mapping gradients
    tech_values = [map_to_tech_space(x, tech_points) for x in test_points]
    inverse_grads = [ForwardDiff.derivative(y -> map_from_tech_space(y, tech_points), y) for y in tech_values]
    
    # Filter out any NaN values
    forward_grads = filter(!isnan, forward_grads)
    inverse_grads = filter(!isnan, inverse_grads)
    
    return forward_grads, inverse_grads
end

function run_tests()
    # Create example technology set
    tech = TechnologySet(
        [1.0, 1.3, 1.8, 2.1],  # A values
        [0.8, 0.6, 0.4, 0.3]   # η values
    )
    
    # Test bijectivity
    max_error, mean_error, std_error = test_bijectivity(tech.A)
    println("Bijectivity test results:")
    println("  Maximum error: ", max_error)
    println("  Mean error: ", mean_error)
    println("  Std error: ", std_error)
    
    # Compute and analyze gradients
    forward_grads, inverse_grads = compute_gradients(tech.A)
    println("\nGradient statistics:")
    println("Forward mapping - mean: ", mean(forward_grads), " std: ", std(forward_grads))
    println("Inverse mapping - mean: ", mean(inverse_grads), " std: ", std(inverse_grads))
    
    # Generate data for plotting
    x_range = range(0, 1, length=100)
    y_tech = [map_to_tech_space(x, tech.A) for x in x_range]
    x_recovered = [map_from_tech_space(y, tech.A) for y in y_tech]
    
    # Generate gradient data with matching dimensions
    x_grad_range = range(0.01, 0.99, length=98)  # Avoid endpoints
    forward_grads_plot = [ForwardDiff.derivative(x -> map_to_tech_space(x, tech.A), x) for x in x_grad_range]
    tech_values_plot = [map_to_tech_space(x, tech.A) for x in x_grad_range]
    inverse_grads_plot = [ForwardDiff.derivative(y -> map_from_tech_space(y, tech.A), y) for y in tech_values_plot]
    
    # Create plots with adjusted margins
    p1 = plot(x_range, y_tech, label="Forward mapping", 
             title="Normalized → Technology Space",
             xlabel="Normalized space", ylabel="Technology space",
             legend=:topright, size=(800, 300), margin=5mm)
    scatter!(p1, [0], [tech.A[1]], label="Min tech")
    scatter!(p1, [1], [tech.A[end]], label="Max tech")
    
    p2 = plot(y_tech, x_recovered, label="Inverse mapping", 
             title="Technology → Normalized Space",
             xlabel="Technology space", ylabel="Normalized space",
             legend=:topright, size=(800, 300), margin=5mm)
    
    # Test smoothness with properly aligned data
    p3 = plot(x_grad_range, forward_grads_plot, label="Forward gradient", 
             title="Mapping Gradients",
             xlabel="Normalized space", ylabel="Gradient",
             legend=:topright, size=(800, 300), margin=5mm)
    plot!(p3, tech_values_plot, inverse_grads_plot, label="Inverse gradient")
    
    # Combine plots
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800,900))
    savefig(final_plot, "mapping_visualization.png")
end

# Run all tests
run_tests() 