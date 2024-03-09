```@meta
EditURL = "../stiff/Orego.jl"
```

````@example Orego
using OrdinaryDiffEq, DiffEqDevTools, ParameterizedFunctions, Plots, ODE, AMF
using LinearAlgebra, StaticArrays

gr() #gr(fmt=:png)

x = @ode_def Orego begin
  dy1 = p1*(y2+y1*(1-p2*y1-y2))
  dy2 = (y3-(1+y1)*y2)/p1
  dy3 = p3*(y1-y3)
end p1 p2 p3

p = SA[77.27,8.375e-6,0.161]
prob = ODEProblem{true, SciMLBase.FullSpecialize}(x,[1.0,2.0,3.0],(0.0,30.0),p)

# generate LU and normal problems
prob, oprob = AMF.generate_trig_amf_prob(prob)

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))

# run solvers with adaptive timestepping.

# LU solves

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats
# 0.042818 seconds
@time sol = solve(prob, ROS34PW1a(; solver_options...), abstol = 1e-12, reltol = 1e-12)
sol.destats
# 18.191901 seconds, maxiters reached
using BenchmarkTools

@benchmark solve($prob, ROS34PW1a(; solver_options...))

# normal solves

@time osol = solve(oprob, KenCarp4());
# 0.000880 seconds
osol.destats

@time osol = solve(oprob, ROS34PW1a(), abstol = 1e-12, reltol = 1e-12)
# 1.581747 seconds
osol.destats


@benchmark solve($oprob, ROS34PW1a())


# Fixed Time steps

using BenchmarkTools

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 1e-1, adaptive = false)

@time sol = solve(prob, ROS34PW1a(; solver_options...), dt= 1e-1, adaptive = false)
#  0.212877 seconds
@benchmark osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)

@time osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)
#   0.001568 seconds

using Literate
Literate.markdown(@__FILE__, joinpath(pwd(), "scripts", "examples", "tutorials"));
nothing #hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

