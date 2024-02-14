cd(dirname(@__DIR__))

using AMF: solve2d, FiniteDifferenceMethod as fdm
using LinearAlgebra

A = 0.1
B = 0.1
C = 0.0
N = 100
init_func = (x, y) -> 16 * x * y * (1 - x) * (1 - y)
function g(du, u, p, t)
    @. du += u^2 * (1 - u) + exp(t)
end

u_exact, sol_exact =
    solve2d(A, B, C, N, g, init_func, fdm(; strategy = "exact_jac", reltol = 1e-14));
u1, sol1 = solve2d(A, B, C, N, g, init_func, fdm(; strategy = "exact_jac", reltol = 1e-8));
u2, sol2 = solve2d(A, B, C, N, g, init_func, fdm(; strategy = "amf_W", reltol = 1e-8));

prob1 = solve2d(
    A,
    B,
    C,
    N,
    g,
    init_func,
    fdm(; strategy = "exact_jac", reltol = 1e-8),
    return_val = "prob",
);
prob2 = solve2d(
    A,
    B,
    C,
    N,
    g,
    init_func,
    fdm(; strategy = "amf_W", reltol = 1e-8),
    return_val = "prob",
);

@show norm(u1 - u_exact) / norm(u_exact) # approx 1e-12
@show norm(u2 - u_exact) / norm(u_exact) # approx 5e-11
