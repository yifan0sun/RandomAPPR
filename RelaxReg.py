import numpy as np
from misc import *
 




def Regularize(laplacian, sigma,  Y):
    n = laplacian.shape[0]
    n_classes = Y.shape[1]

    L = laplacian / (2 * sigma) + np.eye(n) / (2 * n)
    #M = np.linalg.pinv(M)


    for i in range(10):
        try:
            M = np.linalg.pinv(L)
            break
        except:
            L = L + 0.1*np.eye(n)



    track_y = []
    track_psi = []
    track_q = []
    
    G = np.zeros((n_classes, n))

    for t in range(n):
        psi = -2 * (G @ M[:, t])
        q = waterfill(psi)
        ypred = predict(q)
        # g = get_loss_grad(psi, q, Y[t, :])
        G[:, t] = -Y[t, :]

        track_y.append(ypred)
        track_psi.append(psi)
        track_q.append(q)

    track = {
            "y_pred": np.vstack(track_y),
            "psi": np.vstack(track_psi),
            "q": np.vstack(track_q)
        }    
    return track
 

def Relaxation(laplacian, lamb, D, Y):
    
    
    def get_loss_grad(psi, q, y_true):
        if np.dot(y_true, q) == 0:

            i = np.argmax(psi)
            g = np.zeros_like(psi)
            g[i] = 1
            g = g - y_true
            g = g / (1 + 1 / sum(q > 0))

        else:
            qsupport = (q > 0) + 0
            g = -y_true + (qsupport - 1) / np.sum(qsupport)

        return g


    n = laplacian.shape[0]
    n_classes = Y.shape[1]
    L = laplacian / (2 * lamb) + np.eye(n) / (2 * n)
    #M = np.linalg.pinv(M)
    


    for i in range(10):
        try:
            M = np.linalg.pinv(L)
            break
        except:
            L = L + 0.1*np.eye(n)
                            

    track_y = []
    track_psi = []
    track_q = []

    T = np.trace(M)
    A = 0
    G = np.zeros((n_classes, n))

    for t in range(n):
        dem = np.sqrt(A + (D**2) * T)
        psi = -2 * (G @ M[:, t])
        if dem == 0:
            psi *= 0.0
        q = waterfill(psi)

        ypred = predict(q)
        g = get_loss_grad(psi, q, Y[t, :])
        G[:, t] = g
        A = A + 2 * np.dot(g, G @ M[:, t]) + M[t, t] * np.linalg.norm(g) ** 2
        T = T - M[t, t]

        track_y.append(ypred)
        track_psi.append(psi)
        track_q.append(q)

    track = {
        "y_pred": np.vstack(track_y),
        "psi": np.vstack(track_psi),
        "q": np.vstack(track_q)
    }
    return track

 
