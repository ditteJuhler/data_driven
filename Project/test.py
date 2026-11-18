import sympy as ss

S, En, Ei, Ci, I, Q, U, V, N = ss.symbols(
    "S En Ei Ci I Q U V N"
)
Chi1, Chi2, epsilon1, epsilon2, rho, theta, alfa, gamma = ss.symbols(
    "Chi1 Chi2 epsilon1 epsilon2 rho theta alfa gamma"
)

def create_A(S, En, Ei, Ci, I, Q, U, V, N):
    return ss.Matrix([
        [Ci, -Ci, -10*S*I/N, 10*S*I/N - 10*S**2/N, 0, 0, 0, 0, 0, 0],
        [0, 0, 10*S*I/N, -10*S*I/N, -En, 0, 0, 0, 0, 0],
        [0, 0, 0, 10*S*I/N, -Ei, -Ei, Ei, 0, 0, 0],
        [-Ci, 0, 0, 10*S**2/N - 10*S*I/N, 0, 0, 0, 0, 0, 0],
        [0, 0, -I, 0, En, 0, 0, -I, -En, 0],
        [0, 0, 0, 0, Ei, 0, -Ei, I, En, -Q],
        [0, 0, I, 0, 0, 0, 0, 0, 0, Q],
        [0, Ci, 0, 0, 0, Ei, 0, 0, 0, 0],
    ])

# Symbolic parameter vector corresponding to the columns of A
theta_vector = ss.Matrix([
    Chi1,
    Chi1 * epsilon1,
    gamma,
    gamma * rho,
    alfa,
    Chi1 * epsilon2,
    alfa * epsilon2,
    theta,
    alfa * theta,
    Chi2,
])

A_symbolic = create_A(S, En, Ei, Ci, I, Q, U, V, N)


print(A_symbolic @ theta_vector)