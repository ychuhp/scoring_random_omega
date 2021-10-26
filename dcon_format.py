import util
import numpy
import torch

def triu_indicies(N): return [ (a,b) for a,b in zip( *numpy.triu_indices(N,k=1) ) ]

def create_rr(seq,distm, outfp):
    RR = []
    L = distm.shape[0]
    for i,j in triu_indicies(L):
        d = distm[i,j].item()
        if d < 12:
            p =1
            # Alternate probabilities
            # if d < 12: p=1
            # elif d < 16: p= 1 - .5*(16-d)/4
            line = ' '.join([ str(x) for x in [i,j, 0,8, p]])
            RR.append( line )
    util.writeListToFile([seq]+RR, outfp)
    
def ss8_spot1d(fp):
    data = util.writeFileToList(fp)
    return [ x.split()[3] for x in data][1:]
def ss8_alphafold_SS(fp):
    data = util.writeFileToList(fp)
    data = [ x.split() for x in data][1:]
    data = [ x[5] for x in data]
    return data
def ss3_ss8(ss): 
    DCT_ss3_ss8 = {'H':'H', 'E':'E', 'T':'C', 'S':'C', 'G':'H', 'B':'E', 'I':'C', 'C':'C'}
    #Typically, the 8 DSSP states are converted into three classes using the following convention: [GHI] -> h, [EB] -> e, [TS' '] -> c.
    return  [ DCT_ss3_ss8[s] for s in ss]