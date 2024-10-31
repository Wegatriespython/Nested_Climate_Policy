# Vector-Based Portfolio Optimization in Vintage Capital Model

## Problem Statement

### Context
In a vintage capital model with environmental policy, firms must choose how to allocate their investment across different technologies. Each technology is characterized by:
- Its productivity (A): How efficiently it converts inputs to outputs
- Its emissions intensity (η): How much carbon is emitted per unit of output

### Current Limitations
The current model forces firms to choose between a small set of discrete technology options:
- An initial technology
- A current period technology
- A future period technology

This discrete choice structure creates:
1. Unrealistic "all-or-nothing" technology adoption
2. Numerical instability in optimization
3. Poor representation of real-world technology heterogeneity

### Real-World Complexity
In reality, firms face:
- A spectrum of available technologies
- Varying adoption costs across technologies
- Different capabilities to implement technologies
- Path dependencies in technology adoption

## Proposed Solution: Vector-Based Technology Choice

### Core Idea
Instead of choosing between discrete technologies, firms optimize over vectors of technologies:
- Each period has a vector of available technologies (A[1:n], η[1:n])
- Firms choose portfolio weights across these technologies
- Effective technology becomes a weighted average of chosen technologies

### Key Features
1. **Technology Spectrum**
   - Range from low-productivity/low-emissions to high-productivity/high-emissions
   - Captures technology frontier and inferior options
   - Reflects real-world technology heterogeneity

2. **Portfolio Weights**
   - Firms choose fraction of capital for each technology
   - Allows partial adoption of multiple technologies
   - Better represents gradual technology transition

3. **Adjustment Process**
   - Smoother transition between technology states
   - Path-dependent technology adoption
   - Learning and capability building

## Challenges to Address

### 1. Technology Distribution
- How to realistically sample technology vectors?
- What determines technology availability?
- How to maintain economic feasibility?

### 2. Dynamic Evolution
- How do technology vectors evolve between periods?
- What drives technology improvement?
- How to model obsolescence?

### 3. Economic Constraints
- What limits technology adoption?
- How to model adjustment costs?
- What determines minimum efficient scale?

### 4. Computational Feasibility
- How many technologies to include?
- How to maintain tractable optimization?
- What simplifications are acceptable?

## Research Questions

1. **Portfolio Structure**
   - What determines optimal portfolio diversity?
   - How do carbon taxes affect portfolio choice?
   - What role does uncertainty play?

2. **Transition Dynamics**
   - How quickly do firms adjust portfolios?
   - What causes technology lock-in?
   - How do expectations affect transition?

3. **Policy Implications**
   - How do carbon taxes affect technology choice?
   - What policies encourage clean technology adoption?
   - How to prevent stranded assets?

## Next Steps

1. Develop sampling framework for technology vectors
2. Design portfolio optimization structure
3. Implement adjustment cost framework
4. Test computational feasibility
5. Analyze policy implications

*Note: This document outlines the conceptual framework. Implementation details to be developed based on these principles.*