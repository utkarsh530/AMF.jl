@with_kw struct FiniteDifferenceMethod
    strategy::String = "exact_jac"
    solver = ROS34PW1a()
end

function solve2d(A, B, C, N, init_func, alg::FiniteDifferenceMethod; return_prob=false, W_transform=false)
    h = 1 / (N + 1) # homogeneous Dirichlet => can solve linear system on interior grid only.

    # Make finite difference ops
    D = 1/(h^2) * Tridiagonal(ones(N - 1), -2 * ones(N), ones(N - 1))
    # TODO: block diagonal structure, we can specialize much more below than just generic sparsity!
    Dxx = kron(sparse(D), Diagonal(ones(N)))  # TODO: not low bandwidth, needs factorization
                                              # using permutation matrix
    Dyy = kron(Diagonal(ones(N)), sparse(D))
    Dxx_lazy = MatrixOperator(Dxx)
    Dyy_lazy = MatrixOperator(Dyy)

    # Make diffeq op
    op = A * Dxx_lazy + B * Dyy_lazy # TODO: mixed derivative term
    op = cache_operator(op, rand(N^2))
    # op = convert(AbstractMatrix, op)

    # Make ODE function
    function f(du, u, p, t)
        du .= op * u
    end
    
    if alg.strategy == "exact_jac" 
        func = ODEFunction(f; jac_prototype=op)
    elseif alg.strategy in ("exact_W", "amf_W")
        id = IdentityOperator{N^2}() 
        #γ = ScalarOperator(1.0; update_func = (_, _, p, _) -> p) # avoid scalar operator as it is currently broken
        γ = MatrixOperator(Diagonal(ones(N^2)); update_func = (D, _, p, _) -> (D.diag .= p))
        # Wfact below is a lazily composed operator.
        # running factorize on it is clever enough to
        # lazily compose the factorizations of each constituent part.
        # End result: each banded factor is efficiently factorized.
        # this is ugly for a couple reasons:
        # 1) I need to somehow know W_transform in order to set Wfact correctly
        # 2) I couldn't divide by γ at the end (SciMLOperator's threw an error), so duplicated the whole block
        if (!W_transform)
            Wfact_prototype = if alg.strategy == "exact_W"
                -id + γ * op
            elseif alg.strategy == "amf_W"
                -(id - γ * A * Dxx_lazy) * (id - γ * B * Dyy_lazy) 
            end
        else
            Wfact_prototype = if alg.strategy == "exact_W"
                -1/γ + op
            elseif alg.strategy == "amf_W"
                -(id/γ - A * Dxx_lazy) * (id/γ - B * Dyy_lazy) 
            end
        end
        Wfact_prototype = cache_operator(Wfact_prototype, rand(N^2))
        function Wfact(W, u, p, dtgamma, t)
            SciMLOperators.update_coefficients!(γ, u, dtgamma, t)
        end
        func = ODEFunction(f; jac_prototype=op, Wfact_prototype=Wfact_prototype, Wfact=Wfact, Wfact_t=Wfact)
    else
        error("Unsupported strategy.")
    end

    u0 = [init_func(h * i, h * j) for i in 1:N for j in 1:N]
    prob = ODEProblem(func, u0, (0., 1.))
    return_prob && return prob, op, u0 # stop after problem formation, for debugging  

    sol = solve(prob, alg.solver)
    u = sol.u[end]

    # Form 2D solution with BCs padded in
    sol_2d = [zeros(1, N + 2); zeros(N, 1) reshape(u, N, N) zeros(N, 1); zeros(1, N + 2)]

    return sol_2d
end