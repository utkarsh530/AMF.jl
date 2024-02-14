@with_kw struct FiniteDifferenceMethod
    strategy::String = "exact_jac"
    solver = ROS34PW1a
    reltol = 1e-3
end

function solve2d(
    A,
    B,
    C,
    N,
    g,
    init_func,
    alg::FiniteDifferenceMethod;
    return_val = "sol",
    final_t = 1.0,
)
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

    # Make ODE function
    function f(du, u, p, t)
        mul!(du, J_op, u)
        g(du, u, p, t)
        return du
    end

    solver_options = (;)
    if alg.strategy == "exact_jac"
        func = ODEFunction(f; jac_prototype = J_op)
    elseif alg.strategy == "concrete_jac"
        function jac(J, u, p, t)
            update_coefficients!(J_op, u, p, t)
            J .= convert(AbstractMatrix, J_op) # TODO: could be replaced with in-place concretize!
            return J
        end
        func = ODEFunction(f; jac)
    elseif alg.strategy in ("exact_W", "amf_W")
        γ_op = ScalarOperator(
            1.0;
            update_func = (old_val, u, p, t; dtgamma) -> dtgamma,
            accepted_kwargs = (:dtgamma,),
        )
        transform_op = ScalarOperator(
            0.0;
            update_func = (old_op, u, p, t; dtgamma, transform) ->
                transform ? inv(dtgamma) : one(dtgamma),
            accepted_kwargs = (:dtgamma, :transform),
        )

        if alg.strategy == "exact_W"
            W_prototype = -(IdentityOperator(N^2) - γ_op * J_op) * transform_op
        elseif alg.strategy == "amf_W"
            # I would like to write the below two lines, but it doesn't work yet because they won't be concrete and factorizable.
            # W1_op = I - γ_op * J1_op
            # W2_op = I - γ_op * J2_op
            # Instead, I need this hackier code:
            I_N = Diagonal(ones(N))
            _W1_op = MatrixOperator(
                I_N - A * D;
                update_func! = (M, u, p, t; dtgamma) -> (@. M = I_N - dtgamma * A * D),
                accepted_kwargs = (:dtgamma,),
            )
            _W2_op = MatrixOperator(
                I_N - B * D;
                update_func! = (M, u, p, t; dtgamma) -> (@. M = I_N - dtgamma * B * D),
                accepted_kwargs = (:dtgamma,),
            )
            W1_op = Base.kron(_W1_op, IdentityOperator(N))
            W2_op = Base.kron(IdentityOperator(N), _W2_op)

            W_prototype = -(W1_op * W2_op) * transform_op
            solver_options =
                (; solver_options..., linsolve = GenericFactorization(factorize_scimlop))
        end
        W_prototype = cache_operator(W_prototype, zeros(N^2))
        func = ODEFunction(f; jac_prototype = J_op, W_prototype)
    else
        error("Unsupported strategy.")
    end

    u0 = [init_func(h * i, h * j) for i = 1:N for j = 1:N]
    prob = ODEProblem(func, u0, (0.0, final_t))
    solver = alg.solver(; solver_options...)

    if return_val == "prob"
        return prob
    elseif return_val == "integrator"
        return init(prob, solver; reltol = alg.reltol, abstol = nothing)
    elseif return_val == "sol"
        sol = solve(prob, solver; reltol = alg.reltol, abstol = nothing)
        u = sol.u[end]

        # Form 2D solution with BCs padded in
        sol_2d =
            [zeros(1, N + 2); zeros(N, 1) reshape(u, N, N) zeros(N, 1); zeros(1, N + 2)]

        return sol_2d, sol
    elseif return_val == "timing"
        sol = solve(prob, solver; reltol = alg.reltol, abstol = nothing)
        time_solve =
            @belapsed solve($prob, $solver; reltol = $(alg.reltol)) samples = 10 seconds = 1
        return time_solve, sol
    else
        error("Unsupported value $(return_val) for return_val")
    end
end
