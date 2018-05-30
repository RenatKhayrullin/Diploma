from scipy import * # install scipy
from numpy.linalg import norm
from scipy.sparse import *


def create_dence_matrix(size_row, size_col):
    return mat(zeros((size_row, size_col)))


def inverse_matrix(X):
    X.data[:] = 1 / X.data
    return X


def create_initial_matrices(S_L, S_R, S_M, G_Q, G_L, G_R, Y_0, gamma, mu, T, iter_count):
    G_LL = G_L.T * G_L
    G_RR = G_R.T * G_R

    m, n = G_Q.shape
    _, k = G_L.shape

    # initial step
    Y = Y_0.todense().copy()
    C = create_dence_matrix(n, T)
    P_L = create_dence_matrix(k, T)
    P_R = create_dence_matrix(k, T)

    Theta = G_Q * C + G_L * P_L + G_R * P_R

    F = trace(2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R)
    Omega = norm(Y - Theta, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y) + mu * norm(Y - Y_0, ord='fro') ** 2
    objective = F + Omega

    ### Start algorithm #############################################################
    for i in range(iter_count):

        lambda4 = 1 + gamma + mu
        Y = 1 / lambda4 * (gamma * S_M * Y + Theta + mu * Y_0)
        C = 1 / 2 * (S_L * P_L + S_R * P_R + G_Q.T * (Y - G_L * P_L - G_R * P_R))

        P_L = inverse_matrix(identity(G_L.shape[1]) + G_LL) * (S_L.T * C + G_L.T * (Y - G_Q * C - G_R * P_R))
        P_R = inverse_matrix(identity(G_R.shape[1]) + G_RR) * (S_R.T * C + G_R.T * (Y - G_Q * C - G_R * P_L))

        objective_old = objective
        Theta = G_Q * C + G_L * P_L + G_R * P_R
        objective = trace(2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R) + \
                    (norm(Y - Theta, ord='fro') ** 2 + mu * norm(Y - Y_0, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y))

        if (i + 1) % 10 == 0:
            print 'iter', i + 1, 'obj: ', objective, 'rel obj change: ', (objective_old - objective) / objective_old

    Y = G_Q * C + G_L * P_L + G_R * P_R

    return Y, C, P_L, P_R