import sidechainnet as scn
from torch.nn import functional as f
import torch
import numpy as np
from torch.utils.data import Dataset
import os

def int_label_distm(distm):
    '''
    K = 1
    distm = distm*(1/K)
    distm[distm>=35]=35
    return distm.long()
    '''
    x = channel_distm(distm)
    x = x.argmax(dim=1)
    return x

def channel_distm(distm):
    C = 36
    x = torch.stack( C*[distm], axis=1)

    for i in range(C-1):
        x[:,i,:,:] -= i+1

    x = f.relu( -(x.abs()-1) )
    x[:,-1,:,:] = (distm>=C-1)*2

    return x

def unbin_distm(distm):
    ''' undos bin_distM BxCxHxW
    BxHxW
    '''
    #I = 36
    #W = torch.tensor(range(36))
    #return (distm*W[:,None,None]).sum(1).float()
    return torch.argmax(distm, dim=1)

def bin_distm(distm):
    ''' bin_distM HxW
    return CxHxW
    '''
    I = 36
    K = 1
    x = [ (K*(i+1)>distm) & (distm>=K*i) for i in range(I)]
    x.append((distm>=K*I))
    x = torch.stack(x, dim=-1).float()
    return x

def distm_chain(chain):
    ''' Creates a distance matrix
    chain: torch chain of shape BxLxC. B batch, L length, C channels
    returns BxLxL
    '''
    L = chain.shape[1]
    M = torch.stack(L*[chain], axis=1)
    sq_dist = ( M-M.permute([0,2,1,3]) )**2
    distm = torch.sqrt( sq_dist.sum(3) )
    return distm

class dist_dataset(Dataset):

    def __init__(self, mode):
        '''
        mode: train, valid-10,20,30,40,50,70,90, test
        '''
        self.D = scn.load(casp_version=12, thinning=30)
        #self.D = scn.load('debug')
        self.mode = mode

    def __getitem__(self, i):
        mode = self.mode

        seq = self.D[mode]['seq'][i]
        L = len(seq)
        crd = self.D[mode]['crd'][i].reshape(L,14,3)
        #[N, C_alpha, C, Oxygen, side chain coordinates] 0-3, 4-14
        crd = crd[:,1,:]
        distm = distm_chain(crd)
        ang = self.D[mode]['ang'][i]

        return {'seq':seq, 'crd':crd, 'distm':distm, 'ang':ang }

    def __len__(self):
        return len( self.D[mode]['seq'] )

from sidechainnet.utils.sequence import ProteinVocabulary
VOCAB = ProteinVocabulary()
def str_seqint(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(str_seqint(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(lambda c: isinstance(c, str), out)):
        return (None, ''.join(out))

    return out
