using AMF, LinearAlgebra

using SciMLOperators

using OrdinaryDiffEq, LinearSolve


function generate_problems(g, init_func, N; A = 0.1, B = 0.1, C = 0.0, final_t = 1.0)

    h = 1 / (N + 1) # homogeneous Dirichlet => can solve linear system on interior grid only.

    # Make finite difference op
    D = 1 / (h^2) * Tridiagonal(ones(N - 1), -2 * ones(N), ones(N - 1))
    D_op = MatrixOperator(D)

    # Make diffeq op
    @assert iszero(C) "mixed derivative term not handled yet"
    J1_op = A * Base.kron(D_op, IdentityOperator(N))
    J2_op = B * Base.kron(IdentityOperator(N), D_op)
    J_op = J1_op + J2_op
    J_op = cache_operator(J_op, zeros(N^2))

    J = Matrix(J_op)

    function f(du, u, p, t)
        mul!(du, J_op, u)
        g(du, u, p, t)
        return du
    end

    Jupper = (UpperTriangular(J) - Diagonal(J))
    Jlower = (LowerTriangular(J))

    J1__op = MatrixOperator(Jupper)
    J2__op = MatrixOperator(Jlower)

    J_op = J1__op + J2__op
    J_op = cache_operator(J_op, zeros(N^2))

    I_N = Diagonal(ones(N^2))

    @assert Jupper + Jlower == J

    W1_op = MatrixOperator(
        UpperTriangular(zeros(N^2, N^2));
        update_func! = (M, u, p, t; dtgamma) -> (@. M = I_N - dtgamma * Jupper),
        accepted_kwargs = (:dtgamma,),
    )
    W2_op = MatrixOperator(
        LowerTriangular(zeros(N^2, N^2));
        update_func! = (M, u, p, t; dtgamma) -> (@. M = I_N - dtgamma * Jlower),
        accepted_kwargs = (:dtgamma,),
    )

    transform_op = ScalarOperator(
        1e-3;
        update_func = (old_op, u, p, t; dtgamma, transform) ->
            transform ? inv(dtgamma) : one(dtgamma),
        accepted_kwargs = (:dtgamma, :transform),
    )

    W_prototype = -(W1_op * W2_op) * transform_op


    W_prototype = cache_operator(W_prototype, zeros(N^2))

    func = ODEFunction(f; jac_prototype = J_op, W_prototype)
    u0 = [init_func(h * i, h * j) for i = 1:N for j = 1:N]
    oprob = ODEProblem(func, u0, (0.0, final_t))

    exact_prob = ODEProblem(ODEFunction(f; jac_prototype = J_op), u0, (0.0, final_t))

    oprob, exact_prob
end

N = 10

function g(du, u, p, t)
    @. du += u^2 * (1 - u) + exp(t)
end

init_func = (x, y) -> 16 * x * y * (1 - x) * (1 - y)

oprob, exact_prob = generate_problems(g, init_func, N)

solver_options = (; linsolve = GenericFactorization(AMF.factorize_scimlop))
solver = ROS34PW1a(; solver_options...)

@time sol = solve(oprob, solver);

@time exact_sol = solve(exact_prob, ROS34PW1a());

using BenchmarkTools

@belapsed solve($oprob, $solver)

@belapsed solve($exact_prob, $ROS34PW1a())


## Plotting stuff

Ns = [5, 10, 20, 30, 40]

times_amf = Float64[]
times_concrete = Float64[]

for N in Ns
    @info N
    oprob, exact_prob = generate_problems(g, init_func, N)
    t1 = @belapsed solve($oprob, $solver)
    t2 = @belapsed solve($exact_prob, $ROS34PW1a())
    push!(times_amf, t1)
    push!(times_concrete, t2)
end


using Plots

plot(
    Ns .^ 2,
    times_amf,
    marker = :circle,
    yaxis = :log,
    label = "Triangular AMF",
    ylabel = "time (s)",
    xlabel = "# of states in the ODE",
    legend = :topleft,
)
plot!(Ns .^ 2, times_concrete, marker = :circle, label = "Concrete Jacobian")
