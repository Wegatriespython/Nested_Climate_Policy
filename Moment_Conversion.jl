using ForwardDiff
using Statistics
using LinearAlgebra

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
Generate diagonal moments matrix using smooth mapping functions
"""
function generate_diagonal_moments_matrix(sequence_matrix::Matrix{Float64}, period::Int; n_points::Int=100)::Matrix{Float64}
    start_col = period == 0 ? 1 : 4
    A_col = start_col      # A_eff
    η_col = start_col + 1  # η_eff
    c_col = start_col + 2  # cost
    
    # Get unique values for each moment
    A_values = sort(unique(sequence_matrix[:, A_col]))
    η_values = sort(unique(sequence_matrix[:, η_col]))
    c_values = sort(unique(sequence_matrix[:, c_col]))
    
    # Create normalized space points
    x_range = range(0, 1, length=n_points)
    
    # Create diagonal matrix using smooth bijective mapping
    moments_matrix = zeros(Float64, n_points, 3)
    for i in 1:n_points
        x = x_range[i]
        # Map normalized point to technology space using smooth functions
        moments_matrix[i, 1] = map_to_tech_space(x, A_values)
        moments_matrix[i, 2] = map_to_tech_space(x, η_values)
        moments_matrix[i, 3] = map_to_tech_space(x, c_values)
    end
    
    return moments_matrix
end

"""
Generate M_Hole matrix (original discrete points)
"""
function generate_hole_moments_matrix(sequence_matrix::Matrix{Float64}, period::Int)::Matrix{Float64}
    start_col = period == 0 ? 1 : 4
    return sequence_matrix[:, start_col:start_col+2]
end

"""
Generate M_Fixed matrix (continuous mapped points)
"""
function generate_fixed_moments_matrix(sequence_matrix::Matrix{Float64}, period::Int; n_points::Int=100)::Matrix{Float64}
    return generate_diagonal_moments_matrix(sequence_matrix, period, n_points=n_points)
end

"""
Invert moments to find closest matching portfolio using smooth distance metric
"""
function invert_moments_to_portfolio(
    A_eff::Float64, 
    η_eff::Float64, 
    portfolios::Vector{VintagePortfolio}
)::Union{VintagePortfolio, Nothing}
    # Find closest matching portfolio using smooth distance metric
    min_dist = Inf
    best_portfolio = nothing
    
    # Get ranges for normalization
    A_values = Float64[]
    η_values = Float64[]
    for p in portfolios
        A_p, η_p = compute_effective_characteristics(p)
        push!(A_values, A_p)
        push!(η_values, η_p)
    end
    
    # Normalize target points
    A_norm = map_from_tech_space(A_eff, sort(A_values))
    η_norm = map_from_tech_space(η_eff, sort(η_values))
    
    for p in portfolios
        A_p, η_p = compute_effective_characteristics(p)
        # Normalize portfolio points
        A_p_norm = map_from_tech_space(A_p, sort(A_values))
        η_p_norm = map_from_tech_space(η_p, sort(η_values))
        
        # Compute distance in normalized space
        dist = sqrt((A_p_norm - A_norm)^2 + (η_p_norm - η_norm)^2)
        
        if dist < min_dist
            min_dist = dist
            best_portfolio = p
        end
    end
    
    # Return nothing if no close match found (threshold could be adjusted)
    return min_dist > 0.1 ? nothing : best_portfolio
end

"""
Export moments as vector for GE model integration
"""
function export_moments_vector(moments_matrix::Matrix{Float64})::Vector{Vector{Float64}}
    n_points = size(moments_matrix, 1)
    
    A_eff = moments_matrix[:, 1]
    η_eff = moments_matrix[:, 2]
    cost_eff = moments_matrix[:, 3]
    
    return [A_eff, η_eff, cost_eff]
end
