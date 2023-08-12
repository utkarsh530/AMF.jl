function factorize_scimlop(A)
    _fact = LinearAlgebra.factorize(A) 
    # TODO: the input to cache_operator is not constructed generically enough.
    # If factorize_scimlop is provided to GenericFactorization, then this could be solved if GenericFactorization
    # passed along u.
    fact = cache_operator(_fact, zeros(size(A, 2))) 
    return fact
end