
import numpy as np
from misc import *
from appr import appr


def PPR_for_ONL(adj_matrix,degree_vector,beta, sigma, t,num_ops, epsilon, maxneighbors, maxiter):
    n = adj_matrix.shape[0]
    alpha = ((1-beta)*n+sigma)/((1+beta)*n+sigma)
    scale = ((1-beta)*n+sigma)/(2*n*sigma)

    pi,_,_,_,_ = appr(adj_matrix, degree_vector, epsilon, alpha, t, num_ops,maxneighbors , maxiter )
    sd = np.sqrt(degree_vector)
    x = pi / sd
    x = x * sd[t] / scale
    return x
        
def get_trace(adjmat,degrees,beta, sigma,  epsilon, maxneighbors, maxiter,num_ops):
    tr = 0
    n = adjmat.shape[0] 
    for t in range(n):
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t,num_ops, epsilon, maxneighbors, maxiter)
        tr += Mt[t]
    return tr
        
        

def Regularize_APPR(adjmat,degrees,epsilon, beta, sigma,  Y,num_ops, maxneighbors = None, maxiter = None):
    n = adjmat.shape[0]
    n_classes = Y.shape[1] 
    Y = Y.toarray()
    
    track_y = []
    track_psi = []
    track_q = []
    
    G = np.zeros((n_classes, n))
 
    for t in range(n):
        #psi = -2 * (G @ M[:, t])
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t,num_ops, epsilon, maxneighbors, maxiter)
        psi = -2 * (G @ Mt)
        
             
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
 


def Relaxation_APPR(adjmat,degrees,epsilon, beta, sigma, D, Y,num_ops, maxneighbors = None, maxiter = None):
    
    

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


    n = adjmat.shape[0]
    n_classes = Y.shape[1]
    

    track_y = []
    track_psi = []
    track_q = []

    #T = np.trace(M)
    T = get_trace(adjmat,degrees,beta, sigma,  epsilon, maxneighbors, maxiter,num_ops)
    A = 0
    G = np.zeros((n_classes, n))
    for t in range(n):
        dem = np.sqrt(A + (D**2) * T)
        #psi = -2 * (G @ M[:, t])
        
        Mt =  PPR_for_ONL(adjmat,degrees,beta, sigma, t,num_ops, epsilon, maxneighbors, maxiter) 
        psi = -2 * (G @ Mt)
         
                            
        if dem == 0:
            psi *= 0.0
        q = waterfill(psi)

        ypred = predict(q)
        g = get_loss_grad(psi, q, Y[t, :])
        G[:, t] = g
        A = A + 2 * np.dot(g, G @ Mt) + Mt[t] * np.linalg.norm(g) ** 2
        T = T - Mt[t]

        track_y.append(ypred)
        track_psi.append(psi)
        track_q.append(q)

    track = {
        "y_pred": np.vstack(track_y),
        "psi": np.vstack(track_psi),
        "q": np.vstack(track_q)
    }
    return track

 

 
 