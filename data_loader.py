import numpy as np
from scipy.sparse import csr_matrix


def datasets():
    dataset_list = ['political-blog', 'citeseer', 'cora', 'mnist-tr-nei10',
                    'pubmed', 'blogcatalog', 'youtube', 'ogbn-arxiv']
    #dataset_list = ['political-blog', 'citeseer', 'cora','pubmed','mnist-tr-nei10','blogcatalog']
    weighted_dict = {'political-blog': False, 'citeseer': False, 'cora': False,
                     'pubmed': False, 'mnist-tr-nei10': True, 'blogcatalog': False,
                     'youtube': False, 'ogbn-arxiv': False}
    return dataset_list,weighted_dict


def load_graph_data(dataset='citeseer'):
    if dataset not in datasets()[0]:
        print(f"{dataset} does not exist!")
        exit(0)
    f = np.load(f'../../graph_datasets/{dataset}.npz')
    csr_mat = csr_matrix(
        (f["data"], f["indices"], f["indptr"]), shape=f["shape"])
    labels = f['labels']
    return csr_mat, labels


 

def prepare_small_problems(csr_mat,labels):
    d = np.array(csr_mat.sum(axis=1)).flatten()
    Deg = csr_matrix((d, (range(len(d)), range(len(d)))), shape=(len(d), len(d)))
    Theta = Deg - csr_mat
    
    # Calculate the inverse square root of the degree matrix
    D_inv_sqrt = csr_matrix(np.diag(1 / np.sqrt(d)))

    # Compute the normalized Laplacian matrix
    normalized_Theta = D_inv_sqrt @ Theta @ D_inv_sqrt

    
    

    num_classes = int(max(labels) + 1)
    row_indices = np.arange(len(labels))
    col_indices = np.array(labels)
    data = np.ones(len(labels))
    Y_onehot = csr_matrix((data, (row_indices, col_indices)), shape=(len(labels), num_classes))

    
    return Theta,normalized_Theta,  Y_onehot
