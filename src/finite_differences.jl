struct FiniteDifferenceMethod
    do_amf::Bool
end

# type piracy hack to get what I want
function LinearAlgebra.factorize(L::SciMLOperators.AbstractSciMLOperator)
    fact = factorize(convert(AbstractMatrix, L))
    SciMLOperators.InvertibleOperator(fact)
end

function solve2d(A, B, C, N, init_func, alg::FiniteDifferenceMethod; return_prob=false)
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
    
    if !alg.do_amf
        func = ODEFunction(f; jac_prototype=op)
    else
        id = IdentityOperator{N^2}() 
        # Wfact below is a lazily composed operator.
        # running factorize on it is clever enough to
        # lazily compose the factorizations of each constituent part.
        # End result: each banded factor is efficiently factorized.
        γ = ScalarOperator(1.0; update_func = (_, _, p, _) -> p)
        #Wfact_prototype = -id + γ * op
        Wfact_prototype = -(id - γ * A * Dxx_lazy) * (id - γ * B * Dyy_lazy) 
        Wfact_prototype = cache_operator(Wfact_prototype, rand(N^2))
        function Wfact(W, u, p, dtgamma, t)
            SciMLOperators.update_coefficients!(γ, u, dtgamma, t)
        end
        func = ODEFunction(f; jac_prototype=op, Wfact_prototype=Wfact_prototype, Wfact=Wfact)
    end

    u0 = [init_func(h * i, h * j) for i in 1:N for j in 1:N]
    prob = ODEProblem(func, u0, (0., 1.))
    return_prob && return prob, op, u0 # return problem only for debugging

    sol = solve(prob, Rosenbrock23())
    u = sol.u[end]

    # Form 2D solution with BCs padded in
    sol_2d = [zeros(1, N + 2); zeros(N, 1) reshape(u, N, N) zeros(N, 1); zeros(1, N + 2)]

    return sol_2d
end

# Step 1) Efficient linear solves of differential operator

# Step 2) Make prototype of composed operator for W in build function
# Step 3) Update composed operator when W is calc'd 