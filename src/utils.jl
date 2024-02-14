# hack up a factorization that works for SciMLOps with a useful factorize

function factorize_scimlop(A)
    _fact = LinearAlgebra.factorize(A)
    # TODO: the input to cache_operator is not constructed generically enough.
    # If factorize_scimlop is provided to GenericFactorization, then this could be solved if GenericFactorization
    # passed along u.
    fact = cache_operator(_fact, zeros(size(A, 2)))
    return fact
end

function LinearSolve.do_factorization(
    alg::LinearSolve.GenericFactorization{typeof(factorize_scimlop)},
    A,
    b,
    u,
)
    fact = alg.fact_alg(A)
    return fact
end

function LinearSolve.init_cacheval(
    alg::LinearSolve.GenericFactorization{typeof(factorize_scimlop)},
    A::SciMLOperators.AbstractSciMLOperator,
    b,
    u,
    Pl,
    Pr,
    maxiters::Int,
    abstol,
    reltol,
    verbose::Bool,
    assumptions::OperatorAssumptions,
)
    LinearSolve.do_factorization(alg, A, b, u)
end

## generate amf problems with triangular splitting approach
function generate_trig_amf_prob(prob::ODEProblem)

    sys = modelingtoolkitize(prob)

    sys = structural_simplify(sys)

    sys_prob = ODEProblem(sys; jac = true)

    J = calculate_jacobian(sys)

    out, fjac_expr =
        build_function(J, states(sys), parameters(sys), ModelingToolkit.get_iv(sys))

    fjac = @RuntimeGeneratedFunction(fjac_expr)

    J1 = UpperTriangular(J - Diagonal(J))

    out, fjac_upper_expr =
        build_function(J1, states(sys), parameters(sys), ModelingToolkit.get_iv(sys))

    fjac_upper = @RuntimeGeneratedFunction(fjac_upper_expr)

    J2 = LowerTriangular(J)

    out, fjac_lower_expr = build_function(
        J2,
        states(sys),
        parameters(sys),
        ModelingToolkit.get_iv(sys);
        skipzeros = true,
    )
    fjac_lower = @RuntimeGeneratedFunction(fjac_lower_expr)

    M = length(sys_prob.u0)
    Ju = UpperTriangular(zeros(M, M))
    Jl = LowerTriangular(zeros(M, M))
    fjac_upper(Ju, sys_prob.u0, sys_prob.p, 0.0)
    fjac_lower(Jl, sys_prob.u0, sys_prob.p, 0.0)

    J = zeros(M, M)

    fjac(J, sys_prob.u0, sys_prob.p, 0.0)

    @assert Ju + Jl ≈ J

    I_N = Diagonal(ones(M))


    J1_op = MatrixOperator(UpperTriangular(zeros(M, M)); update_func! = fjac_upper)
    J2_op = MatrixOperator(LowerTriangular(zeros(M, M)); update_func! = fjac_lower)
    J_op = J1_op + J2_op

    J_op = cache_operator(J_op, zeros(M^2))

    W1_op = MatrixOperator(
        UpperTriangular(zeros(M, M));
        update_func! = (M, u, p, t; dtgamma) ->
            (fjac_upper(M, u, p, t); @. M = I_N - dtgamma * M),
        accepted_kwargs = (:dtgamma,),
    )
    W2_op = MatrixOperator(
        LowerTriangular(zeros(M, M));
        update_func! = (M, u, p, t; dtgamma) ->
            (fjac_lower(M, u, p, t); @. M = I_N - dtgamma * M),
        accepted_kwargs = (:dtgamma,),
    )

    transform_op = ScalarOperator(
        0.0;
        update_func = (old_op, u, p, t; dtgamma, transform) ->
            transform ? inv(dtgamma) : one(dtgamma),
        accepted_kwargs = (:dtgamma, :transform),
    )
    W_prototype = -(W1_op * W2_op) * transform_op


    W_prototype = cache_operator(W_prototype, zeros(M^2))

    func = ODEFunction(sys_prob.f.f; jac_prototype = J_op, W_prototype)

    amf_prob = ODEProblem(func, sys_prob.u0, sys_prob.tspan, sys_prob.p)
    oprob = ODEProblem(
        ODEFunction(sys_prob.f.f; jac = fjac),
        sys_prob.u0,
        sys_prob.tspan,
        sys_prob.p,
    )
    return amf_prob, oprob
end
