import numpy as np
import matplotlib.pyplot as plt
 

def max_value_indicator(matrix):
    """
    Convert the index of the maximum element in each row to one-hot encoding.

    Args:
    - matrix: 2D array (matrix)

    Returns:
    - one_hot_matrix: 2D array (matrix) with one-hot encoding of maximum element indices in each row
    """
    max_indices = np.argmax(matrix, axis=1)
    num_cols = matrix.shape[1]
    one_hot_matrix = np.zeros_like(matrix)
    rows = np.arange(matrix.shape[0])
    one_hot_matrix[rows, max_indices] = 1
    return one_hot_matrix

def random_indicator_matrix(n, k):
    indicator_matrix = np.zeros((n, k), dtype=int)
    for i in range(n):
        indices = np.random.choice(k, 1, replace=False)
        indicator_matrix[i, indices] = 1
    return indicator_matrix


def one_hot_encode(labels, num_classes):
    """
    Convert a list of labels to their corresponding one-hot encodings.

    Args:
    - labels: List of labels (integers)
    - num_classes: Total number of classes

    Returns:
    - one_hot_matrix: NumPy array containing the one-hot encodings
    """
    one_hot_matrix = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_matrix[i, label] = 1
    return one_hot_matrix


def block_diagonal(matrices):
    num_rows = np.sum([m.shape[0] for m in matrices])
    num_cols = np.sum([m.shape[1] for m in matrices])
    
    block_matrix = np.zeros((num_rows,num_cols))
    
    offsetrow = 0
    offsetcol = 0
    
    for m in matrices:
        r,c = m.shape
        
        block_matrix[offsetrow:(offsetrow+r), offsetcol:(offsetcol+c)] = m
        offsetrow += r
        offsetcol += c
    return block_matrix






def waterfill(r):
    """
    Associated with the optimization problem

    min_{q\in simplex} max_{y\in {1,...,K}} E_{ypred ~ q}[1_{y\neq ypred}] + R(y)

    By writing r such that r_k = R(e_k), we can rewrite this problem as

    min_{q\in simplex} max_{k \in {1,...,K}} (1-q+r)_k

    This can be done through waterfilling.

    1. Start with q = 0
    2. At each iteration, find k = argmax_k (1-q_k+r_k)
    3. Add q_k + tau until this k is no longer the maximum
    4. Repeat steps 2,3 until sum(q) = 1

    """
    n = len(r)
    if np.sum(r) == 0:
        return np.ones(n) / n

    q = np.zeros(n)
    for _ in range(2 * n):
        rq = r - q
        rqmax = np.max(rq)
        S = rq == rqmax
        rq[S] = -np.inf
        k2 = np.argmax(rq)

        tau = rqmax - (r - q)[k2]
        tau = np.minimum(tau, (1 - np.sum(q)) / np.sum(S))
        q[S] += tau
        if np.sum(q) >= 1:
            break
        if np.sum(S) == n:
            tau = (1 - np.sum(q)) / np.sum(S)
            q[S] += tau
            break

    return q


def predict(q):

    if np.sum(q) == 0:
        sampled_index = np.random.choice(len(q))
    else:
        normalized_q = q / np.sum(q)
        sampled_index = np.random.choice(len(q), p=normalized_q)
    y_pred = np.zeros_like(q)
    y_pred[sampled_index] = 1

    return y_pred


def get_running_err(Y_onehot,Y_pred):
    n = Y_onehot.shape[0]

    mistakes = np.sum(Y_pred*(1-Y_onehot),axis=1)
    print(mistakes.shape,n)
    indexwise_misclass = np.zeros(n)
    indexwise_misclass = np.cumsum(mistakes)
    indexwise_misclass = indexwise_misclass / np.array(range(1,n+1))
    
    return indexwise_misclass

 
