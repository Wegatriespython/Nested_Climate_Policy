# Outer Loop Implementation Plan

## 1. State Space Definition
```julia
struct PolicyState
    # Core state variables
    time::Int
    economic_state::Dict{String, Float64}  # From RANK model
    emissions::Float64
    
    # Damage function parameters
    θ_mean::Float64    # Current estimate of damage
    θ_std::Float64     # Uncertainty in estimate
    
    # Policy history
    tax_history::Vector{Float64}
    credibility::Float64  # Derived from policy consistency
end
```

## 2. Policy Action Space
```julia
const TAX_CHANGES = [-0.10, -0.05, 0.0, 0.05, 0.10]  # In percentage points
const MIN_TAX = 0.0
const MAX_TAX = 0.30

struct PolicyAction
    τ_current::Float64
    τ_announced::Float64
end

function get_valid_actions(current_tax::Float64)
    # Generate valid tax combinations within bounds
    return [
        PolicyAction(τ₁, τ₂) 
        for τ₁ in (current_tax .+ TAX_CHANGES) 
        for τ₂ in (τ₁ .+ TAX_CHANGES)
        if MIN_TAX ≤ τ₁ ≤ MAX_TAX && MIN_TAX ≤ τ₂ ≤ MAX_TAX
    ]
end
```

## 3. Damage Function Implementation
```julia
function damage_function(emissions::Float64, θ::Float64)
    # Quadratic damage function
    return θ * emissions^2
end

function update_damage_beliefs(
    θ_prev::Float64, 
    θ_std_prev::Float64, 
    emissions::Float64, 
    output::Float64
)
    # Bayesian update of damage beliefs
    # Returns new (θ_mean, θ_std)
end
```

## 4. Monte Carlo Tree Search
```julia
struct PolicyNode
    state::PolicyState
    action::Union{Nothing, PolicyAction}
    parent::Union{Nothing, PolicyNode}
    children::Vector{PolicyNode}
    visits::Int
    value::Float64
end

function select_action(node::PolicyNode)
    # UCB1 selection with damage uncertainty
end

function expand_node(node::PolicyNode)
    # Generate child nodes for all valid actions
end

function simulate_trajectory(node::PolicyNode)
    # Monte Carlo simulation of one possible future
end

function backpropagate(node::PolicyNode, value::Float64)
    # Update node values up the tree
end
```

## 5. Policy Optimization
```julia
function optimize_policy(current_state::PolicyState, n_iterations::Int)
    root = PolicyNode(current_state, nothing, nothing, [], 0, 0.0)
    
    for _ in 1:n_iterations
        node = select_action(root)
        if node.visits == 0
            value = simulate_trajectory(node)
        else
            expand_node(node)
            child = node.children[1]
            value = simulate_trajectory(child)
        end
        backpropagate(node, value)
    end
    
    return best_action(root)
end
```

## 6. Social Welfare Evaluation
```julia
function evaluate_welfare(
    economic_output::Float64,
    emissions::Float64,
    θ::Float64
)
    # Social welfare = Output - Expected Damages
    damages = damage_function(emissions, θ)
    return economic_output - damages
end
```

## 7. Integration with RANK Model
```julia
function step_environment(
    state::PolicyState,
    action::PolicyAction
)
    # 1. Get RANK equilibrium
    rank_result = compute_equilibrium(action)
    
    # 2. Update emissions
    new_emissions = state.emissions + 
                   rank_result["η_t"] * rank_result["Y_t"]
    
    # 3. Update damage beliefs
    θ_new, θ_std_new = update_damage_beliefs(
        state.θ_mean,
        state.θ_std,
        new_emissions,
        rank_result["Y_t"]
    )
    
    # 4. Update credibility
    new_credibility = update_credibility(
        state.credibility,
        state.tax_history,
        action
    )
    
    # 5. Create new state
    return PolicyState(
        state.time + 1,
        rank_result,
        new_emissions,
        θ_new,
        θ_std_new,
        vcat(state.tax_history, action.τ_current),
        new_credibility
    )
end
```

## 8. Main Loop
```julia
function run_policy_simulation(n_periods::Int)
    # Initialize
    state = initial_policy_state()
    history = [state]
    
    for t in 1:n_periods
        # 1. Optimize policy
        action = optimize_policy(state, 1000)  # 1000 MCTS iterations
        
        # 2. Step environment
        state = step_environment(state, action)
        push!(history, state)
        
        # 3. Log results
        log_period_results(state, action)
    end
    
    return history
end
```

## Implementation Order
1. Basic state and action space implementation
2. Simple damage function and learning
3. Core MCTS without uncertainty
4. Integration with RANK model
5. Add damage uncertainty and learning
6. Full welfare evaluation
7. Analysis tools

## Detailed MCTS Explanation

### How MCTS Works in Climate Policy Context

1. **State Space**
```
State = (Economic Variables, Emissions, Damage Knowledge)
Example State:
- Output (Y) = 19.41
- Emissions (E) = 24.39
- Technology (η) = 1.256
- Current Tax (τ) = 0.05
- Damage Estimate (θ) = 0.002 ± 0.0005
```

2. **Action Space**
```
Actions = Tax Rate Changes (±5,10 basis points)
Example Actions:
(0.05, 0.06) → Current 5%, Announced 6%
(0.06, 0.07) → Current 6%, Announced 7%
```

3. **Tree Search Process**

a) **Selection**
```
UCB1 Score = Average_Welfare + C * sqrt(ln(total_visits)/node_visits)

Example Node:
- Average Welfare = 100
- Visits = 10
- Total Tree Visits = 1000
- Score = 100 + 2 * sqrt(ln(1000)/10) = 104.38
```

b) **Expansion**
```
From State(τ=0.05):
├─ Action(+5bp) → New State
│   ├─ Y = 19.41
│   └─ E = 24.39
├─ Action(+10bp) → New State
└─ Action(-5bp) → New State
```

c) **Simulation (Rollout)**
```
Example Path:
Start: τ=0.05
→ Random: +5bp, Y=19.41, E=24.39
→ Random: +5bp, Y=17.00, E=39.55
→ Random: +0bp, Y=19.18, E=64.17
Total Welfare = 55.59 - Damages(64.17)
```

d) **Backpropagation**
```
Update Path:
Node(τ=0.05):
- Visits: 10 → 11
- Avg Value: 100 → 101.2

Node(τ=0.10):
- Visits: 5 → 6
- Avg Value: 95 → 94.8
```

### Comparison with Backcasting Approach

**1. Backcasting**
```
Advantages:
+ Exhaustive search
+ Guaranteed optimal path
+ Clear interpretation
+ Perfect for fixed targets

Limitations:
- Branching factor = 64^t
- Max ~4 time periods
- No uncertainty handling
- Computationally expensive

Example:
For t=4, paths = 64^4 = 16,777,216
```

**2. MCTS**
```
Advantages:
+ Handles longer horizons
+ Incorporates uncertainty
+ Efficient exploration
+ Adapts to new information

Limitations:
- No guaranteed optimality
- Less interpretable
- Requires parameter tuning

Example:
For t=10, explores ~10,000 promising paths
```

**3. Computational Comparison**
```
Problem: 10-period horizon, 6 possible actions

Backcasting:
- Total paths: 6^10 ≈ 60 million
- Memory needed: ~480GB
- Time: Infeasible

MCTS (1000 iterations):
- Explored paths: ~10,000
- Memory needed: ~80MB
- Time: ~5 minutes
```

### Example MCTS Run

```
Initial State:
- Y₀ = 20.0
- E₀ = 0.0
- θ₀ = 0.001 ± 0.0005

Iteration 1:
1. Select: τ₁ = 0.05 (unexplored)
2. Simulate:
   - Y₁ = 19.41, E₁ = 24.39
   - Y₂ = 17.00, E₂ = 39.55
   Value = 36.41 - 0.001*(39.55)^2

Iteration 2:
1. Select: τ₁ = 0.10 (unexplored)
2. Simulate:
   - Y₁ = 18.50, E₁ = 20.35
   - Y₂ = 16.80, E₂ = 35.20
   Value = 35.30 - 0.001*(35.20)^2
```

### Implementation Considerations

1. **Exploration vs Exploitation**
```julia
C = 2.0  # Higher = more exploration
UCB1_score = average_welfare + C * sqrt(ln(total_visits)/node_visits)
```

2. **Rollout Policy**
```julia
function default_policy(state)
    # Simple heuristic for simulation
    if state.emissions > target
        return random_choice([-0.10, -0.05])
    else
        return random_choice([0.0, 0.05])
    end
end
```

3. **Value Backup**
```julia
function backup(node, value)
    while node !== nothing
        node.visits += 1
        node.value = ((node.visits - 1) * node.value + value) / node.visits
        node = node.parent
    end
end
```

Would you like me to elaborate on any of these components?
