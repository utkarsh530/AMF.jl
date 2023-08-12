using SciMLOperators
using LinearAlgebra
using SparseArrays
using LinearSolve
using IterativeSolvers

# J = J1 + J2
I_minus_J1 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= I - u * u'))
I_minus_J2 = MatrixOperator(rand(2,2); update_func=(mat, u, p, t) -> (mat .= I - u * u' * 2))
γ = ScalarOperator(0.5; update_func=(oldγ, u, p, t; dtgamma) -> dtgamma, accepted_kwargs=(:dtgamma,))
W = I_minus_J1 * I_minus_J2

# user says solve(prob; AMFSolver(jacobian_splitting=(J1, J2)))

W_prototype=W

W = (I - γ * A) * (I - γ * B)

W \ rand(2)

I - γ * J

update_coefficients!(W, rand(2), nothing, nothing; dtgamma=0.7)

(I - γ * A) \ rand(2)

# 

begin
N = 20
D = MatrixOperator(Tridiagonal(rand(N-1), rand(N), rand(N-1)))
# TensorProductOperator(D, I) # need better error
D_tensor_I = TensorProductOperator(D, IdentityOperator(N))
I_tensor_D = TensorProductOperator(IdentityOperator(N), D)

sparse(convert(AbstractMatrix, D_tensor_I + I_tensor_D))
end

W = I - γ * (D_tensor_I + I_tensor_D)
Wapprox = (I - γ * D_tensor_I) * (I - γ * I_tensor_D)

b = rand(400)
@btime solve(D_tensor_I + I_tensor_D, b; alg=KrylovJL_GMRES()) # why does this error?
@btime solve(D_tensor_I, b); solve(I_tensor_D, b);

@btime D_tensor_I \ b
@btime cg(M, b);
M = cache_operator(D_tensor_I + I_tensor_D, b)


MatrixOperator(rand(2,2)) 