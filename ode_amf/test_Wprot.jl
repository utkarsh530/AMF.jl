import Pkg; Pkg.activate(@__DIR__); cd(@__DIR__)

using SciMLOperators
using LinearAlgebra
using SparseArrays
using LinearSolve
using IterativeSolvers
using OrdinaryDiffEq

# J = J1 + J2
J1 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= u * u'))
J2 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= u * u' * 2))
I_minus_J1 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= I - u * u'))
I_minus_J2 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= I - u * u' * 2))
γ = ScalarOperator(0.5; update_func=(oldγ, u, p, t; dtgamma) -> dtgamma, accepted_kwargs=(:dtgamma,))
W = I_minus_J1 * I_minus_J2

f = ODEFunction((u, p, t) -> u; jac_prototype=(J1+J2))
p = ODEProblem(f, [1.0, 1.0], (0.0, 1.0))

sol = solve(prob, Rosenbrock23(autodiff=false)) 

