using OrdinaryDiffEq, Catalyst, ReactionNetworkImporters, Plots, LinearAlgebra, AMF

gr()
# generate ModelingToolkit ODEs
prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "egfr_net.net"))
rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

osys = convert(ODESystem, rn)

## NOTE YET TO TEST BECAUSE NEED TO IMPLEMENT AUTODIFF ALGORITHM

tspan = (0.,15)

prob = ODEProblem{true, SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[], jac = true, sparse = true)

## generate LU and normal problems
prob, oprob = AMF.generate_trig_amf_prob(prob)

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))

## run solvers with adaptive timestepping.

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats
## 0.024140 seconds
@time sol = solve(prob, ROS34PW1a(; solver_options...), abstol = 1e-12, reltol = 1e-12)
sol.destats
## 0.677066 seconds
using BenchmarkTools

@benchmark solve($prob, ROS34PW1a(; solver_options...))

@time osol = solve(oprob, KenCarp4());
## 0.002872 seconds
osol.destats

@time osol = solve(oprob, ROS34PW1a(), abstol = 1e-12, reltol = 1e-12)
## 0.412710 seconds
osol.destats


@benchmark solve($oprob, ROS34PW1a())



## Fixed Time steps

using BenchmarkTools

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 1e-1, adaptive = false)

@time sol = solve(prob, ROS34PW1a(; solver_options...), dt= 1e-1, adaptive = false)
##  0.072238 seconds
@benchmark osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)

@time osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)
##   0.034113 seconds

using Literate
Literate.markdown(@__FILE__, joinpath(pwd(), "scripts", "examples", "tutorials"));

