module AMF

using SciMLOperators
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using MethodOfLines
using ModelingToolkit
using DomainSets
using LinearSolve
using SparseArrays

include("method_of_lines.jl")
include("finite_differences.jl")

export solve2d, FiniteDifferenceMethod, MethodOfLinesMethod

end
