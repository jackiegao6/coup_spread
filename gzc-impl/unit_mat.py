from scipy.io import loadmat
import numpy as np
from scipy import sparse
import networkx as nx


data = loadmat('/root/pythonspace/data-test/datasets/network.mat')
A = data['douban']   # csc_matrix
n, m = A.shape
nnz = A.nnz

density = nnz / (n * m)
sparsity = 1 - density

def analyze_sparse_graph(A):
    n, m = A.shape
    nnz = A.nnz
    density = nnz / (n * m)
    
    symmetric = (A != A.T).nnz == 0
    self_loops = A.diagonal().sum()
    
    return {
        "shape": (n, m),
        "nnz": nnz,
        "density": density,
        "sparsity": 1 - density,
        "symmetric": symmetric,
        "directed": not symmetric,
        "self_loops": self_loops
    }

info = analyze_sparse_graph(A)
for k, v in info.items():
    print(f"{k}: {v}")



if info['directed']:
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
else:
    G = nx.from_scipy_sparse_array(A)

print(G)
