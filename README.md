# MirrorDescent

[![Build Status](https://github.com/ugonj/MirrorDescent.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ugonj/MirrorDescent.jl/actions/workflows/CI.yml?query=branch%3Amain)

Code for the Abstract Mirror Descent method, to minimise abstract convex functions.

## Background

Given a family of functions $L:X→ℝ$, a function is $L$-convex when it can be represented as the supremum of functions from $L$. In this package this is represented by the abstract type `AbstractLinear` and the concrete type `AbstractConvex{T,f}`, respectively, where `T<:AbstractLinear` represents the concrete realisation of abstract linear functions.

The Abstract Mirror Descent method minimises $L$-convex functions.

## User Guide

To use this package, follow these steps:

0. Import the relevant functions from the package.
   ```julia
   import MirrorDescent:Subproblem,minimise
   import Base:+,-,*
   ```
1. Define a concrete realisation of abstract linear functions. For example:
   ```julia
   struct MyLinear
   a
   end

   +(u::MyLinear,v::MyLinear) = MyLinear(u.a+v.a)
   -(u::MyLinear,v::Linear) = MyLinear(u.a-v.a)
   *(s, u::Linear) = MyLinear(s*u.a)
   ```
   can represent all linear functions.

2. Define an $L$-convex function to base the Bregman divergence on. For example:
   ```julia
   @abstract MyLinear Φ(x) = 0.5*x^2
   ```

3. Write an instance of the `minimise` method to solve subproblems from the mirror descent method thus constructed:
   ```julia
   minimise(pb::Subproblem{MyLinear,Val{Φ}}) = -pb.λ.a
   ```

   Note that the subproblem solved at each iteration is the minimisation of $Φ(x) + λ(x)$, where $λ$ is a function from $L$. In our example, the minimiser of the function $0.5 x^2 + ⟨λ,x⟩$ is $-λ$.

4. Define a step size:
  ```julia
  c(k) = 1.0/(k+1)
  ```

5. Define the function you want to minimise and a way to compute its subgradients.
   ```julia
   @abstract Linear f(x) = abs(x)
   sg(x) = Linear(sign(x))
   ```

6. Set up your mirror descent parameters and iterator:
   ```julia
   param = MirrorDescentParameters(Φ,f,c,sg)
   algo = MirrorDescentIteration(param,-2.,Linear(-2))
   ```

7. Finally, run the algorithm, for instance:
   ```julia
   C = collect(Iterators.take(algo,20))
   println(C[end].x) # Solution found after 20 iterations
   ```
