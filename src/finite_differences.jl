@with_kw struct FiniteDifferenceMethod
    strategy::String = "exact_jac"
    solver = ROS34PW1a
    reltol = 1e-3
end

function solve2d(A, B, C, N, g, init_func, alg::FiniteDifferenceMethod; 
                return_val="sol", final_t = 1.)
    h = 1 / (N + 1) # homogeneous Dirichlet => can solve linear system on interior grid only.

    # Make finite difference op
    D = 1/(h^2) * Tridiagonal(ones(N - 1), -2 * ones(N), ones(N - 1))
    D_op = MatrixOperator(D)

    # Make diffeq op
    op = A * (D_op ⊗ I) + B * (I ⊗ D_op) # TODO: mixed derivative term
    op = cache_operator(op, zeros(N^2))

    # Make ODE function
    function f(du, u, p, t)
        mul!(du, op, u)
        g(du, u, p, t)
        return du
    end

    solver_options = (;)
    if alg.strategy == "exact_jac" 
        func = ODEFunction(f; jac_prototype=op)
    elseif alg.strategy == "concrete_jac"
        op_concrete = convert(AbstractMatrix, op)
        jac(J, u, p, t) = (J .= op_concrete)
        func = ODEFunction(f; jac)
    elseif alg.strategy in ("exact_W", "amf_W")
        γ_op = ScalarOperator(1.0; update_func = (u, p, t; dtgamma) -> dtgamma) 
        transform_op = ScalarOperator(0.0;
            update_func = (old_op, u, p, t; dtgamma, transform) -> transform ?
                                                                   inv(dtgamma) :
                                                                   one(dtgamma),
            accepted_kwargs = (:dtgamma, :transform))

        if alg.strategy == "exact_W"
            W_prototype = -(I - γ_op * op) * transform_op
        elseif alg.strategy == "amf_W"
            op1 = I - γ_op * A * (D_op ⊗ I)
            op2 = I - γ_op * B * (I ⊗ D_op)
            W_prototype = -(op1 + op2) * transform_op
            solver_options = (;solver_options..., linsolve=GenericLUFactorization(factorize_scimlop))
        end
        W_prototype = cache_operator(W_prototype, zeros(N^2))
        func = ODEFunction(f; jac_prototype=op, W_prototype)
    else
        error("Unsupported strategy.")
    end

    u0 = [init_func(h * i, h * j) for i in 1:N for j in 1:N]
    prob = ODEProblem(func, u0, (0., final_t))
    solver = alg.solver(; solver_options...)

    if return_val == "prob"
        return prob
    elseif return_val == "integrator"
        return init(prob, solver; reltol=alg.reltol, abstol=nothing)  
    elseif return_val == "sol"
        sol = solve(prob, solver; reltol=alg.reltol, abstol=nothing)
        u = sol.u[end]

        # Form 2D solution with BCs padded in
        sol_2d = [zeros(1, N + 2); zeros(N, 1) reshape(u, N, N) zeros(N, 1); zeros(1, N + 2)]

        return sol_2d, sol
    elseif return_val == "timing" 
        sol = solve(prob, solver; reltol=alg.reltol, abstol=nothing)
        time_solve = @belapsed solve($prob, $solver; reltol=$(alg.reltol)) samples=10 seconds=1
        return time_solve, sol 
    else
        error("Unsupported value $(return_val) for return_val")
    end
end