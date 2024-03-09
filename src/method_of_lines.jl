struct MethodOfLinesMethod end

function solve2d(A, B, C, N, init_func, alg::MethodOfLinesMethod)

    @parameters t x y
    @variables u(..)

    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(y) * Differential(x)

    # 2D PDE with homogeneous Dirichlet boundary conditions
    eq = Dt(u(t, x, y)) ~ A * Dxx(u(t, x, y)) + B * Dyy(u(t, x, y)) + C * Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ init_func(x, y),
        u(t, 0, y) ~ 0,
        u(t, x, 0) ~ 0,
        u(t, 1, y) ~ 0,
        u(t, x, 1) ~ 0,
    ]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = 1 / (N + 1)
    dy = 1 / (N + 1)
    order = 2
    discretization = MOLFiniteDifference([x => dx, y => dy], t; approx_order = order)
    prob = discretize(pdesys, discretization)
    # return prob

    # Solve
    sol = solve(prob, Tsit5())
    u_final = sol[u(t, x, y)][end, :, :]

    return u_final
end
