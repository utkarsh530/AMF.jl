```@meta
EditURL = "../Hires.jl"
```

````@example Hires
using OrdinaryDiffEq, Catalyst, DifferentialEquations, ParameterizedFunctions, ReactionNetworkImporters, Plots, LinearAlgebra, AMF, JLD2

gr() #gr(fmt=:png)

# generate ODE system
f = @ode_def Hires begin
  dy1 = -1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007
  dy2 = 1.71*y1 - 8.75*y2
  dy3 = -10.03*y3 + 0.43*y4 + 0.035*y5
  dy4 = 8.32*y2 + 1.71*y3 - 1.12*y4
  dy5 = -1.745*y5 + 0.43*y6 + 0.43*y7
  dy6 = -280.0*y6*y8 + 0.69*y4 + 1.71*y5 -
           0.43*y6 + 0.69*y7
  dy7 = 280.0*y6*y8 - 1.81*y7
  dy8 = -280.0*y6*y8 + 1.81*y7
end

u0 = zeros(8)
u0[1] = 1
u0[8] = 0.0057

prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,u0,(0.0,321.8122))

# generate LU and normal problems
prob, oprob = AMF.generate_trig_amf_prob(prob)

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))

# run solvers with adaptive timestepping.

# LU solves

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats
# 0.097433 seconds
@time sol = solve(prob, ROS34PW1a(; solver_options...), abstol = 1e-12, reltol = 1e-12)
sol.destats
# 3.902430 seconds
using BenchmarkTools

@benchmark solve($prob, ROS34PW1a(; solver_options...))

# normal solves

@time osol = solve(oprob, KenCarp4());
# 0.001998 seconds
osol.destats

@time osol = solve(oprob, ROS34PW1a(), abstol = 1e-12, reltol = 1e-12)
# 0.122398 seconds
osol.destats


@benchmark solve($oprob, ROS34PW1a())


# Fixed Time steps

using BenchmarkTools

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 1e-1, adaptive = false)

@time sol = solve(prob, ROS34PW1a(; solver_options...), dt= 1e-1, adaptive = false)
#  0.047144 seconds
@benchmark osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)

@time osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)
#   0.012003 seconds

using Literate
Literate.markdown(@__FILE__, joinpath(pwd(), "scripts", "examples", "tutorials"));
nothing #hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

