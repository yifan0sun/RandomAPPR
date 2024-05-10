

import time
import numpy as np
from scipy.sparse import csr_matrix

def find_neighbors(adj_matrix, node_index):
    node_row = adj_matrix[node_index]
    neighbors = node_row.nonzero()[1]
    return neighbors



def push(u,r,x,alpha,A,d, maxneighbors, num_ops):
    
    if d[u] == 0: return
    x[u] = x[u] + alpha * r[u]
    nu = find_neighbors(A,u)
    if maxneighbors is not None:
        if len(nu) > maxneighbors:
            sidx = np.random.choice(len(nu), maxneighbors, replace=False)
            nu = nu[sidx]
    
    
    #r[nu] = r[nu] + (1-alpha)*r[u]/(d[u]*2)*A[nu,u].toarray()[:,0]
    if len(nu) > 0:
        nvec = A[nu,u].toarray()[:,0]
        new_du = np.sum(nvec)
        r[nu] = r[nu] + (1-alpha)*r[u]/(new_du*2)*A[nu,u].toarray()[:,0]
    r[u] = (1-alpha)*r[u]/2
    num_ops[0] += 1 

def appr(A,d,epsilon, alpha, s, num_ops, maxneighbors = None, maxiter = None):
    n = A.shape[0]
    x = np.zeros(n)
    r = np.zeros(n)
    r[s] = 1.
    volr = []
    suppr = []
    iter = 0
    while True:
        iter += 1
        
        if iter % 100 == 0:
            print(iter)
        S = np.where(np.abs(r) > epsilon*d)[0]
        volr.append(len(S))
        suppr.append(np.sum(r>0))
        if len(S) == 0:
            break
        for i in S:
            if r[i] > epsilon * d[i]:
                push(i,r,x,alpha,A,d,maxneighbors, num_ops)
        if maxiter is not None:
            if iter > maxiter:
                break
            
        
    return x,r,volr,suppr, iter
    

def ppr_solve(A,d,alpha,y, timed):
    n = A.shape[0]
    sd = 1/d
    sd[np.isinf(sd)] = 0.
    Theta = ((1+alpha)/2*np.eye(n) - (1-alpha)/2*A.toarray()@np.diag(sd) )/alpha
    #b = np.zeros(n)
    #b[s] = 1
    if timed:    start = time.time()
    x = np.linalg.solve(Theta,y)
    if timed: 
        runtime = time.time() - start
    else: runtime = None
    r = Theta@x - y
    return x,r, runtime


def ppr_getres(A,d,alpha,x,y):
    n = A.shape[0]
    sd = 1/d
    sd[np.isinf(sd)] = 0.
    Theta = ((1+alpha)/2*np.eye(n) - (1-alpha)/2*A.toarray()@np.diag(sd) )/alpha
    #b = np.zeros(n)
    #b[s] = 1
     
    r = Theta@x - y
    return r