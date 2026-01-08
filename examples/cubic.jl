### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 47434909-af8c-4742-b935-614b200c177f
begin
    using Pkg;
	Pkg.add(path="https://github.com/ugonj/MirrorDescent.jl");
	Pkg.add(["Polynomials","Plots","LaTeXStrings","Latexify", "DataFrames"]);
end

# ╔═╡ 46b3f27c-eab9-11f0-93ff-05f7325f8e46
 using MirrorDescent, Polynomials, Plots, LaTeXStrings, Latexify, DataFrames

# ╔═╡ 8115213d-3b69-4495-a60e-68b37a3f0d82
md"""# Cubic functions as abstract linear functions

First we define a cubic function type, which will be the basis for our experiment."""

# ╔═╡ 3ed27452-8c13-4362-9aaf-b728bed792b5
begin
    struct Cubic <: AbstractLinear
	  a # Cubic term
	  b # Linear term
	end

	import Base: +,-,*,show

	(u::Cubic)(x) = u.a*x^3+u.b*x

	+(u::Cubic, v::Cubic) = Cubic(u.a+v.a, u.b+v.b)
	-(u::Cubic, v::Cubic) = Cubic(u.a-v.a, u.b-v.b)
	-(u::Cubic) = Cubic(u.a)
	*(λ,u::Cubic) = Cubic(λ*u.a, λ*u.b)

	function Base.show(io::IO, u::Cubic)
		if(!iszero(u.a)) print(io,u.a, "x³")
			if(!iszero(u.b)) print(io," + ") end
		end
		if(!iszero(u.b)) print(io,u.b, "x") end
	end

    Base.show(io::IO, ::MIME"text/plain", u::Cubic)  =
           print(io, "Cubic:\n   ", u)

	function Base.show(io::IO, ::MIME"text/html", u::Cubic)
        print(io, "<code>Cubic</code>: ")
		if(!iszero(u.a)) print(io,u.a, "x<sup>3</sup>")
			if(!iszero(u.b)) print(io," + ") end
		end
		if(!iszero(u.b)) print(io,u.b, "x") end
		print(io,"\n")
	end

	function Base.show(io::IO, ::MIME"text/plain", f::AbstractConvex{T,S}) where {T,S}
		print(io,"AbstractConvex{$(T)}): L-abstract convex function where L is $(T)")
	end
end

# ╔═╡ 408a60a8-d0b5-442a-847d-7883fa3f3c1f
md"## Objective function"

# ╔═╡ c5d4b78c-803c-402f-9ad0-70056c50c2ab
begin
	Π = [Polynomial([0,-12,0,1]), Polynomial([0,6,0,-1])]
	@abstract Cubic f(x) = maximum(p(x) for p in Π)
end

# ╔═╡ c93ab0f5-9ea4-4c7d-a71f-8bb3ef5622e9
begin
	plot(f,-5,5,linewidth=3,label=nothing)
	#plot!(∇f,linewidth=3,label=L"f'")
	hline!([0], color=:black, lw=1,label=nothing)
	title!(L"f(x)")
end

# ╔═╡ 13328143-867b-4929-a8d5-01697d068371
function subgradient(x)
	c = coeffs(Π[argmax([p(x) for p in Π])])
	return Cubic(c[4],c[2])
end

# ╔═╡ bbd4339a-a839-478b-8280-f5777625a325
md"## Bregman divergence"

# ╔═╡ ac0f8225-6cab-4ce3-a2b8-507ca2eaadee
@latexrun  @abstract Cubic Φ(x) = 3.0/4*x^4 

# ╔═╡ e255c3c6-b5d1-4b9e-a49a-3e0521e002ef
@latexrun ∂Φ(x) = Cubic(x,0)

# ╔═╡ 72a997ce-d342-472b-a292-81adbb49175a
begin
	import MirrorDescent:Subproblem,minimise
	function stationary_points(pb::Subproblem{Cubic,Val{Φ}})
		r = Polynomial([0,pb.λ.b,0,pb.λ.a,3/4])
		return real.(filter(x->isreal(x), roots(derivative(r))))
	end

	function minimise(pb::Subproblem{Cubic,Val{Φ}})
		X = stationary_points(pb)
		return X[argmin(pb.(X))]
	end
end

# ╔═╡ 28307c53-da23-4ec6-bb8f-3f4137dcaabb
begin
	local pb = Subproblem(Φ, Cubic(3.,-1.))
	local X = stationary_points(pb)
	local x = minimise(pb)
	plot(x -> pb(x),-4,4,label="Subproblem objective",linewidth=3)
	scatter!(X,pb.(X),label="Stationnary points")
	scatter!([x],[pb(x)], label="Minimiser")
	hline!([0], color=:black, lw=1,label=nothing)
	hline!([pb(x)], color=:grey,lw=1,label=nothing)
end

# ╔═╡ bfdafc9c-9438-48f0-b21b-b5646ec219f5
md"## Applying the mirror descent method"

# ╔═╡ 40768565-ec56-441a-b7f6-d0bbfe13ef9d
@latexrun c(n) = 1.0/((n+1)^0.6)

# ╔═╡ 0282c21d-1d10-4893-9b23-3ed0a5d9d5b4
param = MirrorDescentParameters(Φ,f,c,subgradient)

# ╔═╡ 303b399a-b0a2-441b-aa10-fbc006d84ba6
begin
	local startingpoints = [-5.0, -0.5 ,0.5, 5.0]
	C = [collect(Iterators.take(MirrorDescentIteration(param,s,∂Φ(s)), 300)) for s in startingpoints];
	[c[end].x for c in C]
end

# ╔═╡ 3a7b7418-5a56-4df5-8cae-8ca471327b10
[findfirst(v->f(v.x)-f(3.)<1e-2, c) for c in C]

# ╔═╡ 06447d55-548f-47c1-859e-b858089bcc5b
begin
	plot(f,-5,5,ylim=(-Inf,60),label=L"f",linewidth=3)
	for v in C[1:1]
		scatter!([c.x for c in v], f.(c.x for c in v), label=latexstring(v[1].x))
	end
	title!("Iterates of the Abstract Mirror Descent algorithm")
	hline!([0], color=:black, lw=1,label=nothing)
end

# ╔═╡ 277b9265-93d4-4bfb-b201-7bd4cacfe87e
begin
	local p = plot()
	for c in C
		plot!([f(v.x)-f(3.) for v in c], linewidth=2, label=latexstring("x_0=$(c[1].x)"))
	end
	xlabel!(L"k"*" (Iterations)")
	ylabel!(L"f(x_k)-f^*")
	#savefig("/tmp/iterations10.png")
	p
end

# ╔═╡ 7f3e95c6-eda5-4355-ab9c-f74e7affe8df
begin
	local p = plot()
	for c in C
		plot!([v.x for v in c], linewidth=2, label=latexstring("x_0=$(c[1].x)"))
	end
	xlabel!(L"k"*" (Iterations)")
	ylabel!(L"x_k")
	#savefig("/tmp/iterations10.png")
	p
end

# ╔═╡ 93495fdd-9cd2-4ff7-8bb4-d60f22c41cbb
md"### Collecting numerical results"

# ╔═╡ e05cb1e5-67a6-43aa-a5e1-5feee5d04ce7
 stepsizes = [:(n->10. / (n+1)), :(n->1. /(n+1)), :(n->1. / (n+1)^0.8)];

# ╔═╡ 34ec0804-4915-434c-ab3a-df60b7249241
cc = eval.(stepsizes);

# ╔═╡ f5e25ac2-835d-4de9-98da-117c43ad8579
begin
	local startingpoints = [-5.0, -0.5 ,0.5, 5.0]
	data = []
	for (cl,c) in zip(stepsizes,cc)
		local param = MirrorDescentParameters(Φ,f,c,subgradient)
		for x₀ in startingpoints
			v = collect(Iterators.take(MirrorDescentIteration(param,x₀,∂Φ(x₀)), 5000))
			local k = findfirst(u->f(u.x)-f(3.)<1e-2, v)
			push!(data,(x₀,latexify(cl),v[100].x,k))
		end
	end
end

# ╔═╡ cb40b441-5791-466c-8984-f2cc9f3cc222
DataFrame(data,[:x₀, :c, :x₁₀₀, :kˢ])

# ╔═╡ Cell order:
# ╠═47434909-af8c-4742-b935-614b200c177f
# ╠═46b3f27c-eab9-11f0-93ff-05f7325f8e46
# ╟─8115213d-3b69-4495-a60e-68b37a3f0d82
# ╠═3ed27452-8c13-4362-9aaf-b728bed792b5
# ╟─408a60a8-d0b5-442a-847d-7883fa3f3c1f
# ╠═c5d4b78c-803c-402f-9ad0-70056c50c2ab
# ╟─c93ab0f5-9ea4-4c7d-a71f-8bb3ef5622e9
# ╠═13328143-867b-4929-a8d5-01697d068371
# ╟─bbd4339a-a839-478b-8280-f5777625a325
# ╠═ac0f8225-6cab-4ce3-a2b8-507ca2eaadee
# ╠═e255c3c6-b5d1-4b9e-a49a-3e0521e002ef
# ╠═72a997ce-d342-472b-a292-81adbb49175a
# ╠═28307c53-da23-4ec6-bb8f-3f4137dcaabb
# ╟─bfdafc9c-9438-48f0-b21b-b5646ec219f5
# ╠═40768565-ec56-441a-b7f6-d0bbfe13ef9d
# ╠═0282c21d-1d10-4893-9b23-3ed0a5d9d5b4
# ╠═303b399a-b0a2-441b-aa10-fbc006d84ba6
# ╠═3a7b7418-5a56-4df5-8cae-8ca471327b10
# ╠═06447d55-548f-47c1-859e-b858089bcc5b
# ╠═277b9265-93d4-4bfb-b201-7bd4cacfe87e
# ╠═7f3e95c6-eda5-4355-ab9c-f74e7affe8df
# ╠═93495fdd-9cd2-4ff7-8bb4-d60f22c41cbb
# ╟─e05cb1e5-67a6-43aa-a5e1-5feee5d04ce7
# ╟─34ec0804-4915-434c-ab3a-df60b7249241
# ╠═f5e25ac2-835d-4de9-98da-117c43ad8579
# ╠═cb40b441-5791-466c-8984-f2cc9f3cc222
