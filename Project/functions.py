import numpy as np

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
def smallpox_A_i(S_i, En_i, Ei_i, Ci_i, I_i, Q_i, U_i, V_i, N):
    A_i = np.array([[Ci_i, -Ci_i, -10 * S_i * I_i / N, 10 * S_i * I_i / N - 10 * S_i**2 / N, 0, 0, 0, 0, 0, 0],
                    [0, 0, 10*S_i*I_i/N, -10*S_i*I_i/N, -En_i, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10*S_i*I_i/N, -Ei_i, -Ei_i, Ei_i, 0, 0, 0],
                    [-Ci_i, 0, 0, 10*S_i**2/N - 10*S_i*I_i/N, 0, 0, 0, 0, 0, 0],
                    [0, 0, -I_i, 0, En_i, 0, 0, -I_i, -En_i, 0],
                    [0, 0, 0, 0, Ei_i, 0, -Ei_i, I_i, En_i, -Q_i],
                    [0, 0, I_i, 0, 0, 0, 0, 0, 0, Q_i],
                    [0, Ci_i, 0, 0, 0, Ei_i, 0, 0, 0, 0]])

    return A_i

