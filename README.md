Contains solvers for the second-order PDE:
$$\frac{\mathrm{d} u}{\mathrm{d} t} = A u_{xx} + B u_{yy} + C u_{xy} + g(y).$$

The main focus is on incorporating Approximate Matrix Factorization (AMF) into the approaches. File
descriptions:

`src/finite_differences.jl`: Implements a spatial discretization approach to the PDE, with an option to use AMF.
`src/method_of_lines.jl`: An alternative implementation to verify correctness. This file hasn't been maintained in a while and is not useed in the analysis. 
`analysis_clean.ipynb`: preliminary analysis of the methods.

## Running code

Getting AMF to work required some changes to the dependencies, so one also needs to check out the following forks:

https://github.com/gaurav-arya/SciMLBase.jl/commits/amf
https://github.com/gaurav-arya/OrdinaryDiffEq.jl/tree/amf
https://github.com/gaurav-arya/SciMLOperators.jl/tree/amf
https://github.com/gaurav-arya/LinearSolve.jl/commits/amf

For each of these libraries, one should clone the repo+branch, and then `dev` the repo in Julia's `Pkg` manager.

