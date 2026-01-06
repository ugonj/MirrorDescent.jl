module MirrorDescent
  import Base:length, iterate, +, -, *
  import Infinities: ℵ₀

  # # Abstract Convex functions
  #
  # Given a family of functions $L:X→ℝ$, a $L$-convex function is a function that can be written as the supremum of 
  # functions from $L$. The functions from $L$ are called *abstract linear*, and $L$-convex functions are also called 
  # *abstract convex*.
  # 
  # We represent the notion of abstract linearity through an abstract type.

  """
      AbstractLinear

  Abstract supertype for all abstract linear functions (eg, `Linear`). All abstract convex functions, defined through `AbstractConvex{L,f}` are parameterised by a subtype `L` of `AbstractLinear`.
  """

  abstract type AbstractLinear end

  # Not sure if this should derive from Function or not?
  """
      AbstractConvex{L,f}

  An abstract convex function, parametrized by a subtype `L` of `AbstractLinear`. Each of these types is a singleton.
  """
  struct AbstractConvex{L,f} <: Function end

  """
      @abstract L f

  Macro to create an `AbstractConvex` function.
  
  # Arguments

  - `L`<:AbstractLinear - the type representing the abstract linear framework.
  - `f` <: Function - a function definition.

  # Example

  ```julia-repl
  julia > @abstract Linear f(x) = 3.5x
  AbstractConvex{Linear,:f}

  julia > f(2.)
  7.0
  ```
  """
  macro abstract(L, f)
      # 1. Deconstruct the function signature
      # f.args[1] is the call part: f(x)
      fcall = f.args[1]
      fname = fcall.args[1]
      fargs = fcall.args[2:end]
      
      # 2. Escape the user-provided type L and arguments
      # We escape L because it's defined in the caller's scope
      esc_L = esc(L)
      esc_args = [esc(a) for a in fargs]
      
      # 3. Construct the new type Expr: AbstractConvex{L, :fname}
      # Note: AbstractConvex must be available in the macro's module
      #typedf = :(AbstractConvex{$esc_L, $(QuoteNode(fname))})
      typedf = :(AbstractConvex{$esc_L, Val{$fname}})
      
      # 4. Define the constructor/alias
      # This makes 'fname' return an instance of the new type
      def = :($(esc(fname)) = $typedf())
      
      # 5. Define the function dispatch
      # We reconstruct: function (f::typedf)(x, y...) ... end
      newcall = Expr(:call, :(f::$typedf), esc_args...)
      newfunc = Expr(:function, newcall, esc(f.args[2]))
      
      return Expr(:block, f, def, newfunc)
  end


  """
      Subproblem{L,T}(Φ,λ)

  Construct a subproblem for the mirror descent method. This is normally used internally by the algorithm.

  The subproblem in the mirror descent is the problem of finding ``y`` such that
  ``Φ(y) - λ(y)`` is minimised, where ``λ`` is an abstract linear function.

  # Arguments
  - `Φ :: AbstractConvex{L,:Φ}` - the function used to define the Bregman Divergence.
  - `λ`::L` - the abstract linear function added to the function to create the subproblem.
  """
  struct Subproblem{L<:AbstractLinear, T}
    Φ::AbstractConvex{L,T1} where T1
    λ::L
  end

  Subproblem(Φ,λ) = Subproblem{typeof(λ),Val{Φ}}(Φ,λ)

  (s::Subproblem{L,T})(x) where {L,T} = s.Φ(x) + s.λ(x)


  """
      MirrorDescentParameters{L,T1,T2}(Φ,f,c[,s])

  The set of parameters for the mirror descent algorithm, which do not change over iterations, namely:

  # Arguments
  - `Φ :: AbstractConvex{L,:Φ}` - the function used to define the Bregman Divergence.
  - `f :: AbstractConvex{L,:f}` - the function used to define the Bregman Divergence.
  - `c <: Function` - a function that returns the step size from the iteration number.
  - `s <: Function` - a function that returns a subgradient of the function `f` at a point `x`.
  """
  struct MirrorDescentParameters{L<:AbstractLinear}
    Φ :: AbstractConvex{L,T1} where T1 # Function to use for the Bregman divergence
    f :: AbstractConvex{L,T2} where T2 # Function to minimise
    c                                  # Step Size
    subgradient                        # A function that returns a ``L``-subgradient of ``f`` at the input.
  end

  # subgradient(f::T) where Tg
  # MirrorDescentParameters(Φ,f,c) = MirrorDescentParameters(Φ,f,c,subgradient(f))

  """
      MirrorDescentIteration{L}(m,x,λ,[u,pb,k])

  An iteration of the abstract mirror descent method for minimising a function `f` at `x`.
  which are ``L``-convex.

  # Fields
  - `m::MirrorDescentParameters`: the parameters for the algorithm.
  - `x` : The current iterate. Set to the initial value in the constructor.
  - `λ::L` : A ``L``-subgradient of the function ``Φ`` at `x`,
  - `u::L` : A ``L``-subgradient of the function ``f``. It is computed directly using the function provided in `m`.
  - `pb::Union{Suproblem{T},Nothing}` :  The subproblem solved at the iteration. `x` is a solution to this subproblem. In the initial step, it is nothing.
  - `k::Integer` : iteration number
  """
  struct MirrorDescentIteration{T<:AbstractLinear}
    m :: MirrorDescentParameters{T}
    "Current iterate"
    x                                              # Current iterate.
    λ :: T                                         # Subgradient of the function Φ at x (selected by the algorithm).
    u :: T                                         # Subgradient of f at x,
    pb :: Union{Subproblem{T,T1},Nothing} where T1 # Subproblem solved at this iteration.
    "Interation number"
    k :: Integer        # Iterate number
  end

  MirrorDescentIteration(m,x,λ) =  MirrorDescentIteration(m,x,λ,m.subgradient(x), nothing, 0)

  Base.length(s::MirrorDescentIteration) = ℵ₀

  Base.iterate(m::MirrorDescentIteration) = (m,m)

  function Base.iterate(m::MirrorDescentIteration, state)
    pb = Subproblem(m.m.Φ, m.m.c(state.k)*state.u - state.λ)
    y = minimise(pb)
    λ = state.λ - m.m.c(state.k)*state.u
    u = state.m.subgradient(y)
    newm = MirrorDescentIteration(m.m,y,λ,u,pb,state.k+1)
    return (newm, newm)
  end

  struct Linear <: AbstractLinear
    a
  end

  coef(u::Linear) = u.a

  +(u::Linear, v::Linear) = Linear(u.a+v.a)
  -(u::Linear) = Linear(-u.a)
  *(l,u::Linear) = Linear(l*u.a)

  Φ(x) = 0.5*x^2

  function minimise(pb::Subproblem{Linear,Val{Φ}})
    return -pb.λ.a
  end

  export MirrorDescentParameters, MirrorDescentIteration, AbstractConvex, AbstractLinear, @abstract, Linear
end
