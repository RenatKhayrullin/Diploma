import numpy as np
import io
from scipy import *
from scipy.sparse import *
from numpy.linalg import norm
from sklearn.preprocessing import normalize


def getMatPos(X):  # sparse matrix or matrix
    X_pos = (abs(X) + X) / 2
    X_neg = (abs(X) - X) / 2
    return X_pos, X_neg


def create_dense_matrix(size_row, size_col):
    return mat(zeros((size_row, size_col)))


def inverse_matrix(X):
    X.data[:] = 1 / X.data
    return X


def normalize_V(V):
    V = mat(normalize(V, norm='l2', axis=0))
    return V


def normalize_U(U):
    U = U / sum(U)
    return U


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def normalize_UV_no_norm(U, V):
    V_colSum = maximum(V.sum(axis=0), 1e-10)
    Q = diags(asarray(V_colSum)[0, :], 0)
    Q_inv = diags(asarray(1.0 / V_colSum)[0, :], 0)
    U = U * Q
    V = V * Q_inv
    return U, V, Q


def normalize_UV(U, V, X_Fnorm):
    X_Fnorm = max(X_Fnorm, 1e-10)
    V_colSum = maximum(V.sum(axis=0), 1e-10)
    Q = diags(asarray(V_colSum / X_Fnorm)[0, :], 0)
    Q_inv = diags(asarray(X_Fnorm / V_colSum)[0, :], 0)
    U = U * Q
    V = V * Q_inv
    return U, V, Q


def calculate_obj_NMF(X, U, V, trXTX):
    U_T_U = U.T * U
    V_T_V = V.T * V
    U_T_X = U.T * X
    obj = sqrt(trXTX + trace(V_T_V * U_T_U) - 2 * trace(U_T_X * V))
    return obj


def NMF(X, U, V, iter_count, tol):
    X_pos, X_neg = getMatPos(X)
    U, V, _ = normalize_UV_no_norm(U, V)
    trXTX = sum((X.T * X).diagonal())
    obj = calculate_obj_NMF(X, U, V, trXTX)

    for iter_count in range(iter_count):
        X_pos_V = X_pos * V
        X_neg_V = X_neg * V
        U_VT_V = U * (V.T * V)
        U = multiply(U, X_pos_V / maximum(U_VT_V + X_neg_V, 1e-10))

        X_pos_T_U = X_pos.T * U
        X_neg_T_U = X_neg.T * U
        UT_U = U.T * U
        V_UT_U = V * UT_U
        V = multiply(V, X_pos_T_U / maximum(V_UT_U + X_neg_T_U, 1e-10))

        U, V, Q = normalize_UV_no_norm(U, V)

        obj_old = obj
        obj = calculate_obj_NMF(X, U, V, trXTX)
        rel = (obj_old - obj) / obj_old

        if abs(rel) < tol:
            return U, V, obj

    return U, V, obj


def calculate_obj(X, U, V, Ustar, alpha, trXTX):
    V_T_V = V.T * V
    U_T_U = U.T * U
    U_T_X = U.T * X
    obj = alpha * norm(U - Ustar, 'fro') + sqrt(trXTX + trace(V_T_V * U_T_U) - 2 * trace(U_T_X * V))
    return obj


def perViewNMF(X, U, V, Ustar, alpha, inner_iter_count, tol):
    trXTX = sum((X.T * X).diagonal())
    Xpos, Xneg = getMatPos(X)

    if issparse(X):
        X_Fnorm = sqrt(sum(X.data) ** 2)
    else:
        X_Fnorm = sqrt(sum(square(X)))
    U, V, Q = normalize_UV(U, V, X_Fnorm)

    obj = calculate_obj(X, U, V, Ustar, alpha, trXTX)

    for iter in range(inner_iter_count):
        XposV = Xpos * V
        XnegV = Xneg * V
        UVTV = U * (V.T * V)
        U = multiply(U, (XposV + alpha * Ustar) / maximum(UVTV + XnegV + alpha * U, 1e-10))

        XposTU = Xpos.T * U
        XnegTU = Xneg.T * U
        UTU = U.T * U
        VUTU = V * UTU
        UstarTU = Ustar.T * U
        V_colSum = V.sum(axis=0)

        tmp1 = XposTU + alpha * tile(UstarTU.diagonal(), (V.shape[0], 1))
        tmp2 = VUTU + XnegTU + alpha * multiply(tile(UTU.diagonal(), (V.shape[0], 1)),
                                                tile(V_colSum[0, :], (V.shape[0], 1)))
        tmp2 = tmp1 / maximum(tmp2, 1e-10)
        del tmp1
        V = multiply(V, tmp2)
        del tmp2

        U, V, Q = normalize_UV(U, V, X_Fnorm)

        obj_old = obj
        obj = calculate_obj(X, U, V, Ustar, alpha, trXTX)
        rel = (obj_old - obj) / obj_old

        if abs(rel) <= tol:
            return U, V, obj

    return U, V, obj


def multiNMF(P_L, U_L, V_L, betaWL, obj_WL,
             P_R, U_R, V_R, betaWR, obj_WR,
             F_s, U_s, V_s, betaFs, obj_Fs,
             alpha, glob_iter_count, inner_iter_count, tol):

    obj_multiNMF = betaWL * obj_WL + betaWR * obj_WR + betaFs * obj_Fs

    for iter in range(glob_iter_count):

        ## update Ustar
        beta_sum = max(betaWL + betaWR + betaFs, 1e-10)
        Ustar = (multiply(betaWL, U_L) + multiply(betaWR, U_R) + multiply(betaFs, U_s)) / beta_sum

        ## update beta
        RE_sum = max(obj_WL + obj_WR + obj_Fs, 1e-10)

        betaWL = -log(obj_WL / RE_sum)
        betaWR = -log(obj_WR / RE_sum)
        betaFs = -log(obj_Fs / RE_sum)

        ## update U and V for each view
        (U_L, V_L, obj_WL) = perViewNMF(P_L, U_L, V_L, Ustar, alpha, inner_iter_count, tol)
        (U_R, V_R, obj_WR) = perViewNMF(P_R, U_R, V_R, Ustar, alpha, inner_iter_count, tol)
        (U_s, V_s, obj_Fs) = perViewNMF(F_s, U_s, V_s, Ustar, alpha, inner_iter_count, tol)

        # calculate obj
        obj_multiNMF_old = obj_multiNMF
        obj_multiNMF = betaWL * obj_WL + betaWR * obj_WR + betaFs * obj_Fs
        rel = (obj_multiNMF_old - obj_multiNMF) / obj_multiNMF_old

        if abs(rel) < tol:
            return U_L, V_L, betaWL, obj_WL, \
                   U_R, V_R, betaWR, obj_WR, \
                   U_s, V_s, betaFs, obj_Fs, \
                   Ustar, obj_multiNMF

    return U_L, V_L, betaWL, obj_WL,\
           U_R, V_R, betaWR, obj_WR,\
           U_s, V_s, betaFs, obj_Fs,\
           Ustar, obj_multiNMF


def clustype_no_clus_algorithm(S_L, S_R, S_M, G_Q, G_L, G_R, Y_0, gamma, mu, iter_count, file_base_name):

    G_LL = G_L.T * G_L
    G_RR = G_R.T * G_R

    m, n = G_Q.shape
    _, k = G_L.shape

    # initial step
    Y = Y_0.todense().copy()
    C = G_Q.T * Y_0.todense()       # create_dense_matrix(n, T)
    P_L = G_L.T * Y_0.todense()     # create_dense_matrix(k, T)
    P_R = G_L.T * Y_0.todense()     # create_dense_matrix(k, T)

    Theta = G_Q * C + G_L * P_L + G_R * P_R

    F = trace(2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R)
    Omega = norm(Y - Theta, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y) + mu * norm(Y - Y_0, ord='fro') ** 2
    objective = F + Omega

    err_rate = file_base_name + "_err.txt"
    with io.open(err_rate, 'a', encoding='utf8') as no_clus_err_rate:

        for i in range(iter_count):
            no_clus_err_rate.write(unicode(str(i) + ";" + str(objective)+"\n"))

            lambda4 = 1 + gamma + mu
            Y = 1 / lambda4 * (gamma * S_M * Y + Theta + mu * Y_0)
            # Y = inverse_matrix(lambda4 * identity(m) - gamma * S_M) * (Theta + mu * Y_0)
            C = 0.5 * (S_L * P_L + S_R * P_R + G_Q.T * (Y - G_L * P_L - G_R * P_R))
            P_L = inverse_matrix(identity(k) + G_LL) * (S_L.T * C + G_L.T * (Y - G_Q * C - G_R * P_R))
            P_R = inverse_matrix(identity(k) + G_RR) * (S_R.T * C + G_R.T * (Y - G_Q * C - G_R * P_L))

            Theta = G_Q * C + G_L * P_L + G_R * P_R

            matrix_F = 2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R
            F = trace(matrix_F)
            Omega = norm(Y - Theta, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y) + mu * norm(Y - Y_0, ord='fro') ** 2
            objective = F + Omega

        Y = G_Q * C + G_L * P_L + G_R * P_R
        no_clus_err_rate.close()

    return Y, C, P_L, P_R


def clustype_algorithm(S_L, S_R, S_M, G_Q, G_L, G_R, Y_0, F_s, F_c,
                       Y_init, C_init, P_L_init, P_R_init,
                       gamma, mu, alpha, T, K, glob_iter_count, inner_iter_count, nmf_iter_count, tol, file_base_name):
    G_LL = G_L.T * G_L
    G_RR = G_R.T * G_R

    m, n = G_Q.shape
    _, k = G_L.shape
    _, ns = F_s.shape
    _, nc = F_c.shape

    # initialize with initial matrices
    Y = Y_init      # dense_matrix(m, T)
    C = C_init      # dense_matrix(n, T)
    P_L = P_L_init  # dense_matrix(k, T)
    P_R = P_R_init  # dense_matrix(k, T)

    # initialize
    # Y = Y_0.todense().copy()
    # C = G_Q.T * Y_0.todense()
    # P_L = G_L.T * Y_0.todense()
    # P_R = G_L.T * Y_0.todense()

    # initialize clustering step matrices
    # normalize_U() -- l1 - normalization
    # normalize_V() -- column l2 - - normalization
    U_L = normalize_U(abs(mat(np.random.rand(k, K))))
    V_L = normalize_V(abs(mat(np.random.rand(T, K))))
    beta_L = 1.0

    U_R = normalize_U(abs(mat(np.random.rand(k, K))))
    V_R = normalize_V(abs(mat(np.random.rand(T, K))))
    beta_R = 1.0

    U_s = normalize_U(abs(mat(np.random.rand(k, K))))
    V_s = normalize_V(abs(mat(np.random.rand(ns, K))))
    beta_Fs = 1.0

    Theta = G_Q * C + G_L * P_L + G_R * P_R
    F = trace(2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R)
    Omega = norm(Y - Theta, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y) + mu * norm(Y - Y_0, ord='fro') ** 2
    objective = F + Omega

    glob_err_rate = file_base_name + "_glob_err.txt"
    for i in range(glob_iter_count):
        with io.open(glob_err_rate, 'a', encoding='utf8') as clus_glob_err_rate:
            clus_glob_err_rate.write(unicode(str(i) + ";" + str(objective)+"\n"))

            lambda4 = 1 + gamma + mu
            Y = 1 / lambda4 * (gamma * S_M * Y + Theta + mu * Y_0)
            # Y = inverse_matrix(lambda4 * identity(m) - gamma * S_M) * (Theta + mu * Y_0)
            C = 1 / 2 * (S_L * P_L + S_R * P_R + G_Q.T * (Y - G_L * P_L - G_R * P_R))
            P_L = inverse_matrix((1 + beta_L) * identity(k) + G_LL) * (
                        S_L.T * C + G_L.T * (Y - G_Q * C - G_R * P_R) + beta_L * U_L * V_L.T)
            P_R = inverse_matrix((1 + beta_R) * identity(k) + G_RR) * (
                        S_R.T * C + G_R.T * (Y - G_Q * C - G_R * P_L) + beta_R * U_R * V_R.T)

            # pre comupute initial approx for U, V
            if i == 0:
                (U_L, V_L, obj_WL) = NMF(P_L, U_L, V_L, nmf_iter_count, tol)
                (U_R, V_R, obj_WR) = NMF(P_R, U_R, V_R, nmf_iter_count, tol)
                (Us_, V_s, obj_Fs) = NMF(F_s, U_s, V_s, nmf_iter_count, tol)

                obj_multiNMF = beta_L * obj_WL + beta_R * obj_WR + beta_Fs * obj_Fs

                objective += obj_multiNMF

                P_L_norm = sqrt(sum(P_L.data) ** 2) if issparse(P_L) else sqrt(sum(square(P_L)))
                U_L, V_L, _ = normalize_UV(U_L, V_L, P_L_norm)

                P_R_norm = sqrt(sum(P_R.data) ** 2) if issparse(P_R) else sqrt(sum(square(P_R)))
                U_R, V_R, _ = normalize_UV(U_R, V_R, P_R_norm)

                F_s_norm = sqrt(sum(F_s.data) ** 2) if issparse(F_s) else sqrt(sum(square(F_s)))
                U_s, V_s, _ = normalize_UV(U_s, V_s, F_s_norm)

            (U_L, V_L, beta_L, obj_WL,
             U_R, V_R, beta_R, obj_WR,
             U_s, V_s, beta_Fs, obj_Fs,
             Ustar, obj_multiNMF) = multiNMF(P_L, U_L, V_L, beta_L, obj_WL,
                                             P_R, U_R, V_R, beta_R, obj_WR,
                                             F_s, U_s, V_s, beta_Fs, obj_Fs,
                                             alpha, inner_iter_count, nmf_iter_count, tol)

            Theta = G_Q * C + G_L * P_L + G_R * P_R
            F = trace(2 * C.T * C + P_L.T * P_L + P_R.T * P_R - 2 * C.T * S_L * P_L - 2 * C.T * S_R * P_R)
            Omega = norm(Y - Theta, ord='fro') ** 2 + gamma * trace(Y.T * Y - Y.T * S_M * Y) + mu * norm(Y - Y_0, ord='fro') ** 2


            objective = F + Omega + obj_multiNMF

            Y = G_Q * C + G_L * P_L + G_R * P_R

    return Y
