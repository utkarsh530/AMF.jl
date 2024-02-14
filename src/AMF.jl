module AMF

using SciMLOperators
using OrdinaryDiffEq
using LinearAlgebra
using LinearSolve
using SparseArrays
using Parameters
using BenchmarkTools
using ModelingToolkit
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

include("utils.jl")
include("finite_differences.jl")

# (MOL baseline commented out for now)

#=
using MethodOfLines
using ModelingToolkit
using DomainSets

include("method_of_lines.jl")
=#

export solve2d, FiniteDifferenceMethod, MethodOfLinesMethod

end
