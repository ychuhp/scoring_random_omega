from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure

from functools import reduce  # Required in Python 3
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


import protein_parser
import data

def distm_pdb(pdb, key=None):
    D = protein_parser.parse(pdb)
    if key==None: key = list(D.keys())[0]
    chain = protein_parser.parse(pdb)[key]['CA']
    chain = torch.tensor([chain])
    chain.shape
    distm = data.distm_chain(chain)
    
    k = data.bin_distm(distm[0])
    k = data.unbin_distm(k.permute(2,0,1).unsqueeze(0))
    return k[0]


from PIL import Image

def fasta_parse(fp):
    ''' Parses a fasta file.
    fp: fasta filepath
    returns: list of ordered pairs (name, sequence)
    '''
    lines = writeFileToList(fp)
    FASTA= []
    first = True
    name,seq = '',''

    for l in lines:
        l = l.strip()
        if l.startswith('>'):
            if first: first = False
            elif not first: FASTA.append( (name,seq) )
            name = l
            seq = ''
        else:
            seq += l
    
    FASTA.append( (name,seq) )
    return FASTA
        


def writeFileToList(filename):
    returnList = []
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            line = line.rstrip('\n')
            returnList.append(line)
    return returnList


def open_image(fp, mode='RGB'):
    '''
    fp: filepath
    mode: RGB, L, LA for rgb, grayscale, grayscale alpha
    '''
    x = Image.open(fp).convert(mode)
    #x = Image.open(fp)
    return torch.tensor( np.array(x) )


def plot_image(x, save=None, scale=8):
    fig = figure(figsize=(scale, scale), dpi=80)
    plt.imshow(x)
    if save: plt.savefig(save, bbox_inches='tight')

    plt.draw()
    plt.show()
    plt.close(fig)

def plot_chain(chain, save=None, mask=None, scale=5):
    L = len(chain)
    if mask is None: mask = L*[1]
    pts = chain

    plt.ion()
    fig = figure(figsize=(scale, scale), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    # for b in barcodes:
    X,Y,Z = [],[],[]

    
    for p,m in zip(chain, mask):
        if m:
            X.append(p[0])
            Y.append(p[1])
            Z.append(p[2])

    ax.plot(X,Y,Z, 'r')

    if save: plt.savefig(save, bbox_inches='tight')

    plt.draw()
    plt.show()
    plt.close(fig)


def save_pred_image(py,y, img_name='test.png'):
    y,py = y.detach().cpu(), py.detach().cpu()
    for a,b in zip(py,y):
        M = torch.triu(torch.ones(py.shape)).bool()
        x = M*py + (~M)*y
        x = torch.cat(list(x), dim=-1)
        plot_image(x, save=img_name)

def save_coord_image(py, y, mask=None, img_name='test_coord.png'):
    y,py = y.detach().cpu(), py.detach().cpu()
    plot_chain(y,save='ychain.png', mask=mask)
    plot_chain(py,save='pychain.png', mask=mask)
    y,py = open_image('ychain.png'), open_image('pychain.png')

    L = y.shape[0] - py.shape[0]
    if L>0: py = torch.nn.functional.pad(py, (0,0,0,0,0,abs(L) ))
    else: y = torch.nn.functional.pad(y, (0,0,0,0,0,abs(L) ))


    x = torch.cat([y,py], dim=1)
    plot_image(x, save=img_name)



# Is a torch-version copy of the function in sklearn.manifold.MDS
def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=0):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distogram is (N x N) and symmetric
        Outs: 
        * best_3d_coords: (batch x N x 3)
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()
    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = [torch.tensor([np.inf]*batch, device=device)]

    # initialize by eigendecomposition: https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
    # follow : https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
    D = pre_dist_mat**2
    M =  0.5 * (D[:, :1, :] + D[:, :, :1] - D) 
    # do loop svd bc it's faster: (2-3x in CPU and 1-2x in GPU)
    # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
    svds = [torch.svd_lowrank(mi) for mi in M]
    u = torch.stack([svd[0] for svd in svds], dim=0)
    s = torch.stack([svd[1] for svd in svds], dim=0)
    v = torch.stack([svd[2] for svd in svds], dim=0)
    best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

    # only eigen - way faster but not weights
    if weights is None and eigen==True:
        return torch.transpose( best_3d_coords, -1, -2), torch.zeros_like(torch.stack(his, dim=0))
    elif eigen==True:
        if verbose:
            print("Can't use eigen flag if weights are active. Fallback to iterative")

    # continue the iterative way
    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        best_3d_coords = best_3d_coords.contiguous()
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[ dist_mat <= 0 ] += 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))

        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        his.append( stress / dis )

    x = torch.transpose(best_3d_coords, -1,-2)#, torch.stack(his, dim=0) ,historic_stresses: (batch x steps)
    x = x.permute(0,2,1)
    return x

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy

from sklearn.manifold import MDS

def mds_coord(distm):
    model = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
    pcoord = model.fit_transform(distm)
    #pcoord = mds_numpy(distm)
    return pcoord



# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/
def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (N_points x Dim).
    """
    #X,Y = torch.tensor(X).float(), torch.tensor(Y).float()
    X,Y = X.permute(1,0), Y.permute(1,0)
    
    device = X.device
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C)
    
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    X_,Y_ = X_.permute(1,0), Y_.permute(1,0)
    return X_, Y_

