@with_kw struct FiniteDifferenceMethod
    strategy::String = "exact_jac"
    solver = ROS34PW1a
    reltol = 1e-3
end

function solve2d(A, B, C, N, g, init_func, alg::FiniteDifferenceMethod; 
                return_val="sol", W_transform=true, final_t = 1.)
    h = 1 / (N + 1) # homogeneous Dirichlet => can solve linear system on interior grid only.

    # Make finite difference op
    D = 1/(h^2) * Tridiagonal(ones(N - 1), -2 * ones(N), ones(N - 1))
    id = Diagonal(ones(N))
    D_op = MatrixOperator(D)
    id_op = oneunit(D)

    # Make diffeq op
    op = A * (D_op ⊗ id_op)  + B * (id_op ⊗ D_op) # TODO: mixed derivative term
    op = cache_operator(op, rand(N^2))

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
        #γ = ScalarOperator(1.0; update_func = (_, _, p, _) -> p) # avoid scalar operator as it is currently broken
        γ_op = MatrixOperator(Diagonal(ones(N^2)); update_func = (D, _, γ, _) -> (D.diag .= γ))
        id2_op = IdentityOperator{N^2}()
        # TODO: I wouldn't have to write my own update funcs below if a SciMLOperator could 
        # be converted into a MatrixOperator with the right update func
        op1 = MatrixOperator(id - A * D; update_func = (M, _, γ, _) -> (@. M = id - γ * A * D)) 
        op2 = MatrixOperator(id - B * D; update_func = (M, _, γ, _) -> (@. M = id - γ * B * D)) 
        # Wfact below is a lazily composed operator.
        # running factorize on it is clever enough to
        # lazily compose the factorizations of each constituent part.
        if alg.strategy == "exact_W"
            Wfact_prototype = -id2_op + γ_op * op
        elseif alg.strategy == "amf_W"
            Wfact_prototype = -(op1 ⊗ op2)
            # MatrixFreeFactorization is a strategy I added to LinearSolve.jl to make a lazy factorization
            # of a lazy operator
            solver_options = (;solver_options..., linsolve=MatrixFreeFactorization())
        end
        # Manually handle W_transform. Not very pretty, ideally the solver would do this for me. 
        W_transform && (Wfact_prototype *= 1/γ_op)
        Wfact_prototype = cache_operator(Wfact_prototype, zeros(N^2))
        # TODO: if update_coefficients! included kwargs, the below could be automated
        function Wfact(W, u, p, dtgamma, t)
            SciMLOperators.update_coefficients!(γ_op, u, dtgamma, t)
            SciMLOperators.update_coefficients!(op1, u, dtgamma, t)
            SciMLOperators.update_coefficients!(op2, u, dtgamma, t)
        end
        func = ODEFunction(f; jac_prototype=op, Wfact_prototype=Wfact_prototype, Wfact=Wfact, Wfact_t=Wfact)
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