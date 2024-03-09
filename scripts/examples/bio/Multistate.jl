using OrdinaryDiffEq, Catalyst, ReactionNetworkImporters, Plots, LinearAlgebra, AMF

gr()
tf       = 20.0

## generate ModelingToolkit ODEs
prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "multistate.net"))

rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

osys = convert(ODESystem, rn)

tspan = (0.,tf)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[])

## generate LU and normal problems
prob, oprob = AMF.generate_trig_amf_prob(prob)

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))

## run solvers with adaptive timestepping.

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats
## 0.004074 seconds
@time sol = solve(prob, ROS34PW1a(; solver_options...), abstol = 1e-12, reltol = 1e-12)
sol.destats
## 2.066134 seconds
using BenchmarkTools

@benchmark solve($prob, ROS34PW1a(; solver_options...))

@time osol = solve(oprob, KenCarp4());
## 0.000269 seconds
osol.destats

@time osol = solve(oprob, ROS34PW1a(), abstol = 1e-12, reltol = 1e-12)
## 0.204009 seconds
osol.destats


@benchmark solve($oprob, ROS34PW1a())

## Fixed Time steps

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 8e-1, adaptive = false)

@time sol = solve(prob, ROS34PW1a(; solver_options...), dt= 8e-1, adaptive = false)
##  0.000992 seconds
@benchmark osol = solve(oprob, ROS34PW1a(), dt = 8e-1, adaptive = false)

@time osol = solve(oprob, ROS34PW1a(), dt = 8e-1, adaptive = false)
##   0.000189 seconds 

using Literate
Literate.markdown(@__FILE__, joinpath(pwd(), "scripts", "examples", "tutorials"));