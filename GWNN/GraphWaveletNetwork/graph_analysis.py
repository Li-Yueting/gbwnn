import numpy as np
import scipy.io
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import sys

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str,alldata = True):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    if(alldata == True):
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally,ty))
        num = labels.shape[0]
        idx_train = range(num/5*3)
        idx_val = range(num/5*3, num/5*4)
        idx_test = range(num/5*4, num)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sort(lamb, U):
    idx = lamb.argsort()
    return lamb[idx], U[:, idx]

def laplacian(W, normalized=False):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

dataset = 'cora'
labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset, alldata=False)

L_normalized = laplacian(adj, normalized=False)
print(L_normalized)
lamb, U = np.linalg.eigh(L_normalized.toarray())
lamb, U = sort(lamb, U)
print('lamb', lamb)
print(features.shape)
F = features.sum(axis=1)

lamb = lamb.reshape([len(lamb), 1])
f = np.matmul(U.T, F)
scipy.io.savemat(str(dataset)+'.mat', {'lambda': lamb, 'f': f, 'F': F})
# threshold = 1e-6
# dataset = pd.DataFrame({'Column1': lamb.T, 'Column2': f.T})
# ax = sns.barplot(x="lambda", y="f", data=dataset, linewidth=2.5)
# plt.hist(f,alpha=0.5,label="work done per worker",color="blue")
# plt.show()
# L_reassemble = U.dot(np.diag(lamb)).dot(U.T)
# L_reassemble[abs(L_reassemble) < threshold] = 0.0
# # scipy.io.savemat('test.mat', {'L_normalized': L_normalized.toarray(), 'L_reassemble': L_reassemble})
# scipy.io.savemat('test.mat', {})
# print(map(float, L_normalized.toarray()))
# print(U.dot(np.diag(lamb)).dot(U.T))