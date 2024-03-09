using Pkg

Pkg.activate(@__DIR__)

const N = 8
const xyd_brusselator = range(0, stop = 1, length = N)
brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a
function brusselator_2d_loop(du, u, p, t)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 =
            limit(i + 1, N), limit(i - 1, N), limit(j + 1, N), limit(j - 1, N)
        du[i, j, 1] =
            alpha *
            (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] - 4u[i, j, 1]) +
            B +
            u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y, t)
        du[i, j, 2] =
            alpha *
            (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] - 4u[i, j, 2]) +
            A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end
p = (3.4, 1.0, 10.0, step(xyd_brusselator))

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end
u0 = init_brusselator_2d(xyd_brusselator)
using OrdinaryDiffEq

prob_ode_brusselator_2d = ODEProblem(brusselator_2d_loop, u0, (0.0, 11.5), p)


using AMF, LinearSolve

@time prob, oprob = AMF.generate_trig_amf_prob(prob_ode_brusselator_2d);

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))

@time sol = solve(prob, KenCarp4(; solver_options...));
sol.destats

@time sol = solve(prob, ROS34PW1a(; solver_options...));
sol.destats

@time osol = solve(oprob, ROS34PW1a());
osol.destats

using BenchmarkTools

@benchmark sol = solve(prob, ROS34PW1a(; solver_options...), dt = 1e-3, adaptive = false)
# BenchmarkTools.Trial: 3 samples with 1 evaluation.
#  Range (min … max):  1.936 s …   2.014 s  ┊ GC (min … max): 1.08% … 1.75%
#  Time  (median):     1.972 s              ┊ GC (median):    1.78%
#  Time  (mean ± σ):   1.974 s ± 39.350 ms  ┊ GC (mean ± σ):  1.57% ± 0.43%

#   █                         █                             █
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   1.94 s         Histogram: frequency by time        2.01 s <

#  Memory estimate: 175.06 MiB, allocs estimate: 1863323.
sol.destats

@benchmark osol = solve(oprob, ROS34PW1a(), dt = 1e-3, adaptive = false)

# BenchmarkTools.Trial: 3 samples with 1 evaluation.
#  Range (min … max):  2.371 s …   2.452 s  ┊ GC (min … max): 0.00% … 2.21%
#  Time  (median):     2.375 s              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.399 s ± 45.846 ms  ┊ GC (mean ± σ):  0.75% ± 1.28%

#   █ █                                                     █
#   █▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   2.37 s         Histogram: frequency by time        2.45 s <

#  Memory estimate: 38.94 MiB, allocs estimate: 80576.

osol.destats
