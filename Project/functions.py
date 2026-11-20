import numpy as np
from scipy.interpolate import CubicSpline


# Define the Smallpox model
def smallpox_model(t, x, p):
    # p = [Chi1, Chi2, epsilon1, epsilon2, rho, theta, alfa, gamma]
    # x = states [S, En, Ei, Ci, I, Q, U, V]
    
    Chi1, Chi2, epsilon1, epsilon2, rho, theta, alfa, gamma, N = p
        
    S, En, Ei, Ci, I, Q, U, V = x
    
    phi = I/S
    beta = 10 * gamma /(phi * N)
    
    
    dS = Chi1 * (1 - epsilon1) * Ci - beta * (phi + rho - phi * rho) * S * I
    dEn = beta * phi * (1 - rho) * S * I - alfa * En
    dEi = beta * phi * rho * S * I - (Chi1 * epsilon2 + alfa * (1 - epsilon2)) * Ei
    dCi = beta * rho * (1 - phi) * S * I - Chi1 * Ci
    dI = alfa * (1 - theta) * En - (theta + gamma) * I
    dQ = alfa * (1 - epsilon2) * Ei + theta * (alfa * En + I) - Chi2 * Q
    dU = gamma * I + Chi2 * Q
    dV = Chi1 * (epsilon2 * Ei + epsilon1 * Ci)
    
    dxdt = np.array([dS, dEn, dEi, dCi, dI, dQ, dU, dV])

    return dxdt


""" Linear system functions """
def smallpox_theta(par):
    return np.array([par[0], par[0]*par[2], par[7], par[7]*par[4], par[6],
                     par[0]*par[3], par[6]*par[3], par[5], par[6]*par[5], par[1]])


# Creating a function h that takes theta and returns omega: w = h(theta)
def smallpox_h(theta):
    omega = np.array([theta[0], theta[9], theta[1]/theta[0], theta[5]/theta[0], theta[3]/theta[2], theta[7], theta[4], theta[2]])
    return omega

# Linear system matrix
def smallpox_A(y, N):
    S, En, Ei, Ci, I, Q, U, V = y
    A = np.array([[Ci, -Ci, -10 * S * I / N, 10 * S * I / N - 10 * S**2 / N, 0, 0, 0, 0, 0, 0],
                    [0, 0, 10*S*I/N, -10*S*I/N, -En, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10*S*I/N, -Ei, -Ei, Ei, 0, 0, 0],
                    [-Ci, 0, 0, 10*S**2/N - 10*S*I/N, 0, 0, 0, 0, 0, 0],
                    [0, 0, -I, 0, En, 0, 0, -I, -En, 0],
                    [0, 0, 0, 0, Ei, 0, -Ei, I, En, -Q],
                    [0, 0, I, 0, 0, 0, 0, 0, 0, Q],
                    [0, Ci, 0, 0, 0, Ei, 0, 0, 0, 0]])

    return A

""" Begin regression algorithm """

# Integrate linear model to create J

def J(t, y, N, matrix_fn, shape_J):
    A_list = []
    
    for i in range(len(t)):
        y_i = np.asarray(y[:, i])
        A_i = matrix_fn(y_i, N)
        
        A_list.append(A_i)

    A_values = np.stack(A_list, axis=0)

    assert shape_J == A_values.shape, "The shape of A must match the shape of J (m, n, p)"

    A_spline = CubicSpline(t, A_values)

    J = np.zeros(shape_J)
    for i in range(len(t)):
        J[i] = A_spline.integrate(t[0], t[i])

    return J

# Created the "augmented" system (??)
def M(J):
    m, n, p = J.shape
    I_n = np.eye(n)
    I_n_stacked = np.broadcast_to(I_n, (m, n, n))

    M = np.concatenate((J, I_n_stacked), axis=2)
    return M.reshape((n*m, p+n))

