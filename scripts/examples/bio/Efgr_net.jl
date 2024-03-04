using OrdinaryDiffEq, Catalyst, ReactionNetworkImporters, Plots, LinearAlgebra, AMF

gr()
# generate ModelingToolkit ODEs
prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "egfr_net.net"))
rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

osys = convert(ODESystem, rn)

## NOTE YET TO TEST BECAUSE NEED TO IMPLEMENT AUTODIFF ALGORITHM

tspan = (0.,15)

prob = ODEProblem{true, SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[])

using ForwardDiff
u  = ModelingToolkit.varmap_to_vars(nothing, species(rn); defaults=ModelingToolkit.defaults(rn))
du = copy(u)
p  = ModelingToolkit.varmap_to_vars(nothing, parameters(rn); defaults=ModelingToolkit.defaults(rn))

J = AbstractArray{Float64, 356}
J = ForwardDiff.jacobian!(J, (du, u) -> f!(du, u, p ,t), du, u)


#prob, oprob = AMF.generate_trig_amf_prob(prob)

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats

@time sol = solve(prob, ROS34PW1a(; solver_options...), abstol = 1e-12, reltol = 1e-12)
sol.destats

using BenchmarkTools

@benchmark solve($prob, ROS34PW1a(; solver_options...))

oprob = ODEProblem(
    ODEFunction(sys_prob.f.f; jac = fjac),
    sys_prob.u0,
    sys_prob.tspan,
    sys_prob.p,
)

@time osol = solve(oprob, KenCarp4());
osol.destats

@time osol = solve(oprob, ROS34PW1a(), abstol = 1e-12, reltol = 1e-12)
osol.destats


@benchmark solve($oprob, ROS34PW1a())

using BenchmarkTools

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 1e-1, adaptive = false)


@benchmark osol = solve(oprob, ROS34PW1a(), dt = 1e-1, adaptive = false)

