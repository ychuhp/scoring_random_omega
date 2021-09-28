# -*- coding: utf-8 -*-
__author__ = "Yechan Hong"
__maintainer__ = "Yechan Hong"
__email__ = "ychuh@pharmcadd.com"
__status__ = "Dev"


from collections import defaultdict
import numpy as np


DICT_R2RES = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D','CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K','MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
    'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V','ASX':'N', 'GLX':'Q', 'UNK':'G', 'HSD':'H',
    }

def process_raw_line(l,mode='PDB'):
    ''' Given a raw ATOM line in a PDB extract the information of interest
    l: raw line
    mode: PDB/CIF
    returns: chain_id, chain idx, RESIDUE, atom type, x,y,z
    '''
    if mode=='PDB':
        return process_raw_line_pdb(l)
    elif mode=='CIF':
        return process_raw_line_cif(l)
    raise Exception('Invalid mode specified.')

def process_raw_line_pdb(l):
    ''' Given a raw ATOM line in a PDB extract the information of interest
    l: raw line
    returns: chain_id, chain idx, RESIDUE, atom type, x,y,z
    '''
    chain_id = l[20:22].strip()
    idx = l[22:26].strip()
    RES = l[17:20].strip()
    atom = l[12:16].strip()
    x,y,z = float(l[30:38]), float(l[38:46]), float(l[46:54])

    return chain_id, int(idx), RES, atom, x,y,z

def process_raw_line_cif(l):
    ''' Given a raw ATOM line in a CIF extract the information of interest
    l: raw line
    returns: chain_id, chain idx, RESIDUE, atom type, x,y,z
    '''
    # Courtesy of Manuel Alessandro Collazo
    l = l.split()
    chain_id = l[18], 
    idx, RES, atom = l[8], l[5], l[3]

    res_idx = l.index(RES,5+1)
    chain_idx = res_idx+1
    chain_id = l[chain_idx]

    x,y,z = float(l[10]), float(l[11]), float(l[12])
    return chain_id, int(idx), RES, atom, x,y,z

def cb(ca,c,n):
    ''' Compute the coordinates of cb 
    ca: np coordinates of ca
    c: np coordinates of c
    n: np coordinates of n
    returns: coordinates of cb
    '''
    ca,c,n = np.array(ca), np.array(c), np.array(n)
    b,c = ca-n, c-ca
    a = np.cross(b,c)
    return (-0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca).tolist()


def dihedral(a,b,c,d):
    a,b,c,d = np.array(a), np.array(b), np.array(c), np.array(d)

    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)

def angle(a,b,c):
    a,b,c,d = np.array(a), np.array(b), np.array(c), np.array(d)

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)

def isatomline(l,mode='PDB'):
    ''' Validation check
    l: pdb or cif line
    mode: PDB/CIF
    returns Boolean if the given line is a valid ATOM line
    '''
    if mode == 'PDB':
        return isatomline_pdb(l)
    elif mode == 'CIF':
        return isatomline_cif(l)
    raise Exception('Invalid mode specified.')


def isatomline_pdb(l):
    ''' Validation check
    l: pdb ATOM line
    returns: Boolean if the given line is indeed a valid ATOM line
    '''
    if not l.startswith('ATOM '):
        return False
    return True

def isatomline_cif(l):
    ''' Validation check
    l: cif ATOM line
    returns: Boolean if the given line is indeed a valid ATOM line
    '''
    if not l.startswith('ATOM '):
        return False
    l = l.split()
    
    if not len(l)>19: return False
    RES = l[5]
    ATOM = l[3]
    try:
        res_idx = l.index(RES,5+1) # residual match check
        atom_idx = l.index(ATOM,3+1) # atom match check
        x,y,z = float(l[10]), float(l[11]), float(l[12]) # float x,y,z check
    except ValueError as e:
        return False

    if not RES == l[res_idx]: #RESIDUE
        return False
    if not ATOM == l[atom_idx]: #ATOM TYPE
        return False
    return True

def parse(filepath, mode='PDB', atoms = ['CA', 'N', 'C', 'CB'] ):
    ''' Parses a given cif and constructs a dictionary of CA coordinates and sequence information based on 
    filepath: filepath of cif
    mode: PDB/CIF
    returns: Fdata a dictionary indexed by chain_ids where each chain_id is a dictionary of 'SEQ'uence and 'CA'-coordinates
    '''
    fr = open(filepath)
    lines = fr.readlines()

    #
    # First Parse: build structured data
    #
    Sdata = defaultdict( lambda:defaultdict(dict) )
    old_idx = -1
    for l in lines:
        chain_id, idx = None, None

        if isatomline(l, mode):
            chain_id, idx, RES, atom, x,y,z = process_raw_line(l, mode)
            if RES in DICT_R2RES:
                Sdata[chain_id][idx]['type'] = DICT_R2RES[RES]
                Sdata[chain_id][idx][atom] =  [x,y,z]

        if not (chain_id and idx): continue
        # If we move to a increasing new index and the previous residue does not have CB, check if we can construct and assign the CB to the previous residue
        ores = Sdata[chain_id][old_idx]
        if idx > old_idx and all(a in ores for a in ['CA','C','N']) and ('CB' not in ores):
            ores['CB'] = cb(ores['CA'], ores['C'], ores['N']) 

        old_idx = idx

    # 
    # Second Parse: format and construct from structured data
    #
    Fdata = defaultdict( lambda:defaultdict(list) )
    for chain in Sdata:
        CA,CB,C,N = [],[],[],[]
        seq = ''
        for idx in Sdata[chain]:
            r = Sdata[chain][idx]
            if all(a in r for a in atoms):
                Fdata[chain]['SEQ'].append( r['type'] )
                [ Fdata[chain][a].append(r[a]) for a in atoms ]

        Fdata[chain]['SEQ'] = ''.join( Fdata[chain]['SEQ'] )

    return Fdata
