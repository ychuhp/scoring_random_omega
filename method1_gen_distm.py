import util
import torch
import json
import argparse
import os
import dist_map_score as dms
import torch
from os import path
import math

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, help="predicted pdb file", required=True)
    parser.add_argument('--pred_error', type=str, help="predicted aligned error json file", required=True)
    parser.add_argument('--out', type=str, help="output folder", required=True)
    args = parser.parse_args()


    os.mkdir( args.out )

    # Load json pae file
    with open(args.pred_error) as json_file:
        data = json.load(json_file)


    e = torch.tensor( data[0]['distance'] )
    L = int(math.sqrt( len(e) ))
    e = e.reshape(L,L)
    e = e + e.T

    #Load direct pae file
    '''
    e = torch.load(args.pred_error)
    e = torch.tensor(e)
    e = e + e.T
    '''

    distm = util.distm_pdb(args.pdb)

    torch.save( e<10, args.out + '/confidence_mask' )
    torch.save( distm, args.out + '/dist' )
