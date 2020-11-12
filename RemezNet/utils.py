import pickle as pk

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg.eigen.arpack import eigsh


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def gen_data(opt='', data=2):
    if data == 1:
        graph = nx.read_edgelist('../data/america_revo', nodetype=int, comments='%')
    elif data == 2:
        graph = nx.read_weighted_edgelist('../data/lang_country', comments='%')
    elif data == 3:
        graph = nx.read_weighted_edgelist('../data/crime', comments='%')

    data_num = len(graph.nodes)

    norm_lap = nx.normalized_laplacian_matrix(graph)
    if opt == 'gcn':
        norm_lap = normalize(nx.adjacency_matrix(graph) + sp.eye(nx.adjacency_matrix(graph).shape[0]))

    features = np.ones((len(graph.nodes), 1))

    train_rate = 0.6
    val_rate = 0.7
    test_rate = 1.

    data_ind = np.arange(data_num)
    np.random.shuffle(data_ind)

    idx_train = sorted(data_ind[range(int(data_num * train_rate))])
    idx_val = sorted(data_ind[range(int(data_num * train_rate), int(data_num * val_rate))])
    idx_test = sorted(data_ind[range(int(data_num * val_rate), int(data_num * test_rate))])

    labels = pk.load(open('../data/{}_labels.pk'.format(data), 'rb'))

    return norm_lap, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cheby2poly(params):
    params_tup = tuple(map(tuple, params))[0]
    cheb = np.polynomial.chebyshev.Chebyshev(params_tup)
    coef = np.polynomial.chebyshev.cheb2poly(cheb.coef)
    return coef


def customized_loss(X, y):
    loss = torch.sum(torch.pow(X - y, 2))
    return loss


def max_loss(X, y):
    loss = torch.max(torch.pow(X - y, 2))
    return loss


def rat_func(x, outs_nu, outs_de):
    p_len = len(outs_nu)
    q_len = len(outs_de)
    xp = [x ** i for i in range(p_len)]
    xq = [x ** i for i in range(q_len)]
    px = np.dot(outs_nu, np.array(xp))
    qx = np.dot(outs_de, np.array(xq))

    return px / qx


def poly_func(x, outs_nu):
    p_len = len(outs_nu)
    xp = [x ** i for i in range(p_len)]
    px = np.dot(outs_nu, np.array(xp))

    return px


def rat_func_str(outs_nu, outs_de):
    p_len = len(outs_nu)
    q_len = len(outs_de)
    xp = "+".join(["{}x^{}".format(outs_nu[i], str(i)) for i in range(p_len)])
    xq = "+".join(["{}x^{}".format(outs_de[i], str(i)) for i in range(q_len)])

    return "({})/({})".format(xp, xq)


def poly_func_str(outs_nu):
    p_len = len(outs_nu)
    xp = "+".join(["{}x^{}".format(outs_nu[i], str(i)) for i in range(p_len)])

    return xp


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def poly_recur(support, orders=4, poly_style='norm', lap_eig='eig'):  # norm or cheby polynomials
    t_k = list()
    if lap_eig == 'eig':
        t_k.append(np.ones(support.shape[0]))
    elif lap_eig == 'lap':
        t_k.append(np.eye(support.shape[0]))
    t_k.append(support)

    def normal_recurrence(support, t_k_minus_one):
        if lap_eig == 'eig':
            return np.multiply(support, t_k_minus_one)
        elif lap_eig == 'lap':
            return np.matmul(support, t_k_minus_one)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, support):
        if lap_eig == 'eig':
            return 2 * np.multiply(support, t_k_minus_one) - t_k_minus_two
        elif lap_eig == 'lap':
            return 2 * np.matmul(support, t_k_minus_one) - t_k_minus_two

    for i in range(2, orders + 1):
        if poly_style == 'cheby':
            t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], support))
        elif poly_style == 'norm':
            t_k.append(normal_recurrence(t_k[-1], support))

    return np.array(t_k)


def rational(x, c, m, n):
    p1 = 0.0
    for i in range(m + 1):
        p1 = p1 + c[i] * (x ** i)
    q1 = 1.0
    for i in range(m + 1, n + m + 1):
        q1 = q1 + c[i] * (x ** (i - m))
    return p1 / q1


def func(x, opt=5):
    if opt == 0:
        return np.sqrt(x)
    elif opt == 1:
        return abs(x - .5) * .5 + .5 * x
    elif opt == 2:
        return np.minimum(abs(x - .5), np.exp(x))
    elif opt == 3:
        return np.sign(x - .5)
    elif opt == 4:
        return np.sqrt(abs(x - .5))
    elif opt == 5:
        return abs(x - .5)
    elif opt == 6:
        return x / (abs(x - .5) + 0.1) / 10
    elif opt == 7:
        return np.sqrt(x)
    elif opt == 8:
        return np.maximum(.5, np.sin(x + x ** 2)) - x / 20
    elif opt == 9:
        return -x - x ** 2 + np.exp(-(30 * (x - .5)) ** 2)
    elif opt == 10:
        return np.sign(x - .2) - np.sign(x - .8)
    elif opt == 11:
        return 1 / (1 + x ** 2)
    elif opt == 12:
        return np.exp(x)
    elif opt == 13:
        return np.log(x)
    elif opt == 14:
        return np.sin(x)
    elif opt == 15:
        return x ** 2


def rational_lap(lap, c, m, n):
    p = []
    for i in range(m + 1):
        p.append(c[i] * (lap ** i))

    q = [sp.eye(lap.shape[0])]
    for i in range(m + 1, n + m + 1):
        q.append(c[i] * (lap ** (i - m)))

    p_m, q_m = sum(p), np.linalg.pinv(np.sum(q).A)

    return p_m.dot(q_m)


def extract_rational(rat, x):
    de = rat.weight_de.data.cpu().numpy()
    nu = rat.weight_nu.data.cpu().numpy()
    c = np.concatenate((nu, de), axis=1)[0]

    return rational(x, c, rat.m_orders, rat.n_orders)


def extract_rational_lap(rat, lap):
    de = rat.weight_de.data.cpu().numpy()
    nu = rat.weight_nu.data.cpu().numpy()
    c = np.concatenate((nu, de), axis=1)[0]

    return rational_lap(lap, c, rat.m_orders, rat.n_orders)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']

    plt.plot(x, y1, label='D_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = np.os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)
    plt.close()


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(new_x):
        return pointwise(new_x).tolist()

    return ufunclike


if __name__ == '__main__':
    gen_data()
