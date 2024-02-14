cd(dirname(@__DIR__))

using AMF: solve2d, FiniteDifferenceMethod as fdm
import AMF
using Plots
using LinearAlgebra
using SciMLOperators
using LinearSolve
using JLD2
using DiffEqDevTools

### Setup PDE

A = 0.1
B = 0.1
C = 0.0
N = 100
init_func = (x, y) -> 16 * x * y * (1 - x) * (1 - y)
function g(du, u, p, t)
    @. du += u^2 * (1 - u) + exp(t)
end

# kwargs: strategy, solver, rtol
function run_job(N = N; return_val = "sol", final_t = 1, kwargs...)
    return solve2d(A, B, C, N, g, init_func, fdm(; kwargs...); return_val, final_t)
end

### Collect linear solve timings

function linsolve_stats(N, strategy)
    time_solve, sol = run_job(N; strategy, return_val = "timing")
    time_per_linsolve = time_solve / sol.destats.nsolve
    iterations_per_linsolve = sol.destats.nw / sol.destats.nsolve # only meaningful for iterative methods
    return (; time_per_linsolve, nsolve = sol.destats.nsolve, iterations_per_linsolve)
end

function collect_linsolve_data(Ns)
    amf = []
    exact = []
    concrete = []
    for N in Ns
        @info "Now trying..." N
        push!(amf, (N, linsolve_stats(N, "amf_W")))
        push!(exact, (N, linsolve_stats(N, "exact_W")))
        (N <= 30) && push!(concrete, (N, linsolve_stats(N, "concrete_jac")))
    end
    return amf, exact, concrete
end

function get_linsolve_data()
    amf, exact, concrete = collect_linsolve_data(10:10:30)
    amf2, exact2 = collect_linsolve_data(40:20:140)

    amf_full = vcat(amf, amf2)
    concrete_full = concrete
    exact_full = vcat(exact, exact2)
    return amf_full, concrete_full, exact_full
end

data = get_linsolve_data()
save("plots/linsolve_data.jld2", Dict("data" => data))

function plot_linsolve_data(amf, concrete, exact)
    p = plot(
        yaxis = :log,
        xlabel = "N",
        ylabel = "Time per linear solve",
        legend = :topleft,
    )
    plot!(
        p,
        first.(amf),
        map(d -> d.time_per_linsolve, last.(amf)),
        label = "AMF",
        markershape = :circle,
    )
    plot!(
        p,
        first.(exact),
        map(d -> d.time_per_linsolve, last.(exact)),
        label = "Krylov method",
        markershape = :circle,
    )
    plot!(
        p,
        first.(concrete),
        map(d -> d.time_per_linsolve, last.(concrete)),
        label = "Gaussian elimination",
        markershape = :circle,
    )
    plot!(p, dpi = 1000)
    plot!(p, yticks = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    savefig(p, "plots/timing.png")
    return p
end

p = plot_linsolve_data(data...)

### Make AMF accuracy diagram

function get_amf_errors()
    integrator_exact = run_job(100; strategy = "exact_W", return_val = "integrator")
    integrator_amf = run_job(100; strategy = "amf_W", return_val = "integrator")
    step!(integrator_exact)
    step!(integrator_amf)
    W_exact = integrator_exact.cache.W
    W_amf = integrator_amf.cache.W
    linsolve_exact = integrator_exact.cache.linsolve
    linsolve_amf = integrator_amf.cache.linsolve

    γs = []
    matvec_errors = []
    inv_errors = []

    for γ_log = -7:1:0
        γ = 10.0^(γ_log)
        @info "Next γ..." γ
        update_coefficients!(
            W_exact,
            nothing,
            nothing,
            nothing;
            dtgamma = γ,
            transform = false,
        )
        update_coefficients!(
            W_amf,
            nothing,
            nothing,
            nothing;
            dtgamma = γ,
            transform = false,
        )
        u = rand(100^2)
        matvec_exact = W_exact * u
        matvec_amf = W_amf * u
        push!(γs, γ)
        push!(matvec_errors, norm(matvec_exact - matvec_amf) / norm(matvec_exact))
        linsolve_exact.b = copy(u)
        linsolve_amf.b = copy(u)
        linsolve_exact.A = W_exact
        linsolve_amf.A = W_amf
        inv_exact = solve(linsolve_exact; reltol = 1e-14).u
        inv_amf = solve(linsolve_amf; reltol = 1e-14).u
        @show norm(inv_amf)
        push!(inv_errors, norm(inv_exact - inv_amf) / norm(inv_exact))
    end

    return γs, matvec_errors, inv_errors
end

γs, matvec_errors, inv_errors = get_amf_errors()

function plot_amf_errors(γs, matvec_errors, inv_errors)
    p = plot(yticks = (10.0) .^ (-7:1:0), xaxis = :log, yaxis = :log, markershape = :circle)
    plot!(p, yticks = (10.0) .^ (-7:1:0), label = "Matrix-vector product")
    plot!(p, xlabel = "γ", ylabel = "Relative error")
    plot!(p, γs[1:7], γs[1:7] .^ 2 * 1e7, label = "Quadratic scaling", linestyle = :dash)
    plot!(
        p,
        γs,
        matvec_errors,
        label = "Matrix-vector product",
        legend = :topleft,
        markershape = :circle,
    )
    plot!(p, γs, inv_errors, label = "Inversion", markershape = :circle)
    plot!(p, dpi = 800)
    savefig(p, "plots/amferrors.png")
    return p
end

plot_amf_errors(γs, matvec_errors, inv_errors)

#### Plot exact solution

exact_sol, _ = run_job(100; strategy = "exact_jac", return_val = "sol", reltol = 1e-14);
p = heatmap(exact_sol)
plot!(p, dpi = 800)
savefig(p, "plots/exact.png")

### Make work-precision diagram

function get_workprec_data(rtols_log = (-1:-1:-10))
    reltols = []
    errors_w = []
    errors_amf = []
    time_solves_w = []
    time_solves_amf = []
    solns_w = []
    solns_amf = []

    for rtol_log in rtols_log
        @info "Next rtol..." rtol = "1e$(rtol_log)"
        reltol = 10.0^(rtol_log)
        push!(reltols, reltol)
        sol_w, soln_w = run_job(N; strategy = "exact_W", return_val = "sol", reltol)
        @info "W solved"
        sol_amf, soln_amf = run_job(N; strategy = "amf_W", return_val = "sol", reltol)
        @info "AMF solved"
        push!(errors_w, norm(sol_w - exact_sol) / norm(exact_sol))
        push!(errors_amf, norm(sol_amf - exact_sol) / norm(exact_sol))
        push!(solns_w, soln_w)
        push!(solns_amf, soln_amf)
        time_solve_w, _ = run_job(N; strategy = "exact_W", return_val = "timing", reltol)
        @info "W timed"
        time_solve_amf, _ = run_job(N; strategy = "amf_W", return_val = "timing", reltol)
        @info "AMF timed"
        push!(time_solves_w, time_solve_w)
        push!(time_solves_amf, time_solve_amf)
    end
    return reltols, errors_w, errors_amf, time_solves_w, time_solves_amf, solns_w, solns_amf
end

data = get_workprec_data(-1:-1:-10)
save("plots/workprec_data.jld2", Dict("data" => data))

function plot_workprec_data(data)
    reltols, errors_w, errors_amf, time_solves_w, time_solves_amf, solns_w, solns_amf = data
    p = plot(
        yaxis = :log,
        xaxis = :log,
        xlabel = "Relative error",
        ylabel = "Time (s)",
        legend = :topright,
        xticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
    )
    plot!(p, errors_w, time_solves_w, label = "Krylov method", markershape = :circle)
    plot!(p, errors_amf, time_solves_amf, label = "AMF-W", markershape = :circle)
    plot!(p, dpi = 800)
    savefig(p, "plots/workprec.png")
    return p
end

plot_workprec_data(data)

### Form pseudotime integrator

# Set up pseudotime problem

N = 100
init_func = (x, y) -> 0 # start at all 0's

soln_func(x, y) = -16 * x * y * (1 - x) * (1 - y)  # the previous init function is now our goal!
soln_discretized = [soln_func(i / (N + 1), j / (N + 1)) for i = 1:N for j = 1:N]
goal(x, y) = A * (-32y - 32x + 32x^2 + 32y^2) # the hand-calculated Laplacian of the goal
goal_discretized = [goal(i / (N + 1), j / (N + 1)) for i = 1:N for j = 1:N]
function g(du, u, p, t)
    @. du += goal_discretized
end

function get_pseudotime_data()
    integrator_amf = run_job(N; strategy = "amf_W", return_val = "integrator")
    J = integrator_amf.cache.J

    iters_amf = []
    errs_amf = []
    iters_krylov = []
    errs_krylov = []
    for reltol_log = -1:-1:-13
        @info "Considering reltol 1e$(reltol_log)"
        reltol = 10.0^(reltol_log)

        # solve via AMF-W
        exact_sol, soln =
            run_job(N; strategy = "exact_W", return_val = "sol", reltol, final_t = 100000)
        num_iters_amf = soln.destats.nsolve
        err_amf =
            norm(exact_sol[2:N+1, 2:N+1][:] - soln_discretized) / norm(soln_discretized)

        push!(iters_amf, num_iters_amf)
        push!(errs_amf, err_amf)

        # solve via Krylov
        prob = LinearProblem(-J, goal_discretized)
        krylov_sol = solve(prob; reltol, maxiters = 1000, abstol = 1e-30)
        num_iters_krylov = krylov_sol.iters
        err_krylov = norm(krylov_sol - soln_discretized) / norm(soln_discretized)

        push!(iters_krylov, num_iters_krylov)
        push!(errs_krylov, err_krylov)
    end

    # Remove points where AMF has already reached max convergence
    iters_amf = iters_amf[1:6]
    errs_amf = errs_amf[1:6]

    return iters_amf, errs_amf, iters_krylov, errs_krylov
end

data = get_pseudotime_data()
save("plots/data.jld2", Dict("data" => data))

function plot_pseudotime_data(data)
    iters_amf, errs_amf, iters_krylov, errs_krylov = data
    p = plot(xlabel = "Relative error", ylabel = "Iterations", xaxis = :log)
    plot!(p, errs_krylov, iters_krylov, markershape = :circle, label = "Krylov method")
    plot!(p, errs_amf, iters_amf, markershape = :circle, label = "Pseudotime AMF-W")
    plot!(p, dpi = 700)
    savefig(p, "plots/pseudotime.png")
    return p
end

plot_pseudotime_data(data)
