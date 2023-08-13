# hack up a factorization that works for SciMLOps with a useful factorize

function factorize_scimlop(A)
    _fact = LinearAlgebra.factorize(A) 
    # TODO: the input to cache_operator is not constructed generically enough.
    # If factorize_scimlop is provided to GenericFactorization, then this could be solved if GenericFactorization
    # passed along u.
    fact = cache_operator(_fact, zeros(size(A, 2))) 
    return fact
end

function LinearSolve.do_factorization(alg::LinearSolve.GenericFactorization{typeof(factorize_scimlop)}, A, b, u)
    fact = alg.fact_alg(A)
    return fact
end 

function LinearSolve.init_cacheval(alg::LinearSolve.GenericFactorization{typeof(factorize_scimlop)}, A::SciMLOperators.AbstractSciMLOperator, b, u, Pl, Pr, maxiters::Int, abstol,
    reltol, verbose::Bool, assumptions::OperatorAssumptions)
    LinearSolve.do_factorization(alg, A, b, u)
end