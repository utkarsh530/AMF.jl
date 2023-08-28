Contains solvers for the second-order PDE:
$$\frac{\mathrm{d} u}{\mathrm{d} t} = A u_{xx} + B u_{yy} + C u_{xy} + g(u).$$

The main focus is on incorporating Approximate Matrix Factorization (AMF) into the approaches. File
descriptions:

`src/finite_differences.jl`: Implements a spatial discretization approach to the PDE, with an option to use AMF.

`src/method_of_lines.jl`: An alternative implementation to verify correctness. This file hasn't been maintained in a while and is not useed in the analysis. 

`scripts/minimal.jl`: minimal example.

`scripts/analysis_clean.jl`: preliminary analysis of the methods.

## Running scripts

Open Julia in this directory and run the following in Julia's package manager: 
```julia
] activate scripts
] dev .
] instantiate
```
in Julia's REPL package manager, which activates the environment in the `scripts` directory, develops the `src` code in the `AMF.jl` module here, and then builds the environment. The scripts should now be runnable in this environment.

