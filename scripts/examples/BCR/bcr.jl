using DiffEqBase,
    OrdinaryDiffEq,
    Catalyst,
    ReactionNetworkImporters,
    Sundials,
    Plots,
    DiffEqDevTools,
    ODEInterface,
    ODEInterfaceDiffEq,
    LSODA,
    TimerOutputs,
    LinearAlgebra,
    ModelingToolkit,
    BenchmarkTools,
    LinearSolve

gr()
datadir = joinpath(dirname(pathof(ReactionNetworkImporters)), "../data/bcr")
const to = TimerOutput()
tf = 100000.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng =
    loadrxnetwork(BNGNetwork(), joinpath(datadir, "bcr.net"))
show(to)
rn = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = convert(ODESystem, rn)
show(to)

tspan = (0.0, tf)
@timeit to "ODEProb No Jac" oprob =
    ODEProblem{true,SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[])

@timeit to "ODEProb Jac" oprob_jac =
    ODEProblem{true,SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[]; jac = true)

show(to)
oprob_sparse = ODEProblem{true,SciMLBase.FullSpecialize}(
    osys,
    Float64[],
    tspan,
    Float64[];
    sparse = true,
);
