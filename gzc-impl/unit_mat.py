from scipy.io import loadmat
import numpy as np
from scipy import sparse
import networkx as nx

'''
dict_keys(['__header__', '__version__', '__globals__', 'ans', 'Airtraffic', , 'Amazonrandom', 'Bitcoin', 'CAGrQc', 'CAHepTh',
'CAHepTh1core', 'CAHepThrandom', 'DNCemail', 'Digg', 'Diggrandom', 'EmailEnron', 'EmailEnron1core', 'EmailEnron9core', 'EmailEnronrandom',
'Epinions', 'Epinions16core', 'Epinionsrandom', 'Haggle', 'Infectious', 'Italypowergrid', 'PDZBase', 'USpowergrid', 'WikiVote',
'WikiVote1core', 'WikiVote2core', 'WikiVoterandom', 'as733', 'as7332', 'ascaida', 'doubanrandom',
'driversize', 'elegans', 'euroad', 'facebook', 'facebookego1core', 'googlehyperlink', 'googlehyperlinkrandom', 'hunmanprotein',
'jazz', 'maayan', 'mat', 'moreno', 'netAstrophysics', 'netCatster', 'netDog', 'netRealityMining', 'netRouteview', 'netSistercity',
, 'netactorcollaboration', 'netarenasemail', 'netascaida', 'netasskitter', 'netbrightkite', 'netciteseer',
'netdbpediasimilar', 'neteuroroad', 'netfacebookego', 'netflickr', 'netgenefusion', 'nethamster', 'nethyves', 'netlivemocha',
'netorkut', 'netreactome', 'netscience', 'networksize', 'nodes', 'oregon', 'oregon1core', 'oregon2core', 'oregonradom', 'p2pGnutella',
'petster', 'powergrid', 'reactome', 'reactome1core', 'roundworm', 'socEpinions', 'socsign', 'youtubefriendship', 'youtubefriendshiprandom',
'f', 'f1', 'spreadrate', 'step', 'net', 'Haggle2', 'oldtag', 'as73322', 'DNCemail2', 'hunmanprotein2', 'Bitcoin2', 'k', 'threshold',
 'averageeigenvalue', 'eigenvalues', 'lambda', 'row', 'vol', 'j', 'averagGq',
'giantfraction', 'averagLT', 'averageeigenvalue2', 'averagLT2', 'SF40', 'degreedist', 'SF35', 'SF30', 'SF25', 'SF20', 'ER'])
'''

# 'Amazon', 'douban', 'douban11core', 'netYeast', 'youtubefriendship', 'doubanrandom', 'netactorcollaboration'
data = loadmat('/root/pythonspace/data-test/datasets/network.mat')
A = data['netactorcollaboration']   # csc_matrix
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


