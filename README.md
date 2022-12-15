Contains solvers for the second-order PDE:
$$\frac{\mathrm{d} u}{\mathrm{d} t} = A u_{xx} + B u_{yy} + C u_{xy} + g(y).$$

The main focus is on incorporating Approximate Matrix Factorization (AMF) into the approaches.

## Running code

Getting AMF to work required some changes to the dependencies, so one also needs to check out the following forks:

https://github.com/gaurav-arya/SciMLBase.jl/commits/amf
https://github.com/gaurav-arya/OrdinaryDiffEq.jl/tree/amf
https://github.com/gaurav-arya/SciMLOperators.jl/tree/amf
https://github.com/gaurav-arya/LinearSolve.jl/commits/amf



