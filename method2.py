import argparse
import os
import dist_map_score as dms
import util
import torch
from os import path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, help="the folder that contains the pdb files", required=True)
    args = parser.parse_args()




    distm_fp = 'distm/dist'
    mask_fp = 'distm/confidence_mask'
    pdb_dir = args.pdb_dir

    A = torch.load(distm_fp)
    mask = torch.load(mask_fp)


    for pdb_fp in os.listdir(pdb_dir):

        if pdb_fp[-4:] != '.pdb': continue
        B = util.distm_pdb( path.join(pdb_dir,pdb_fp) ).clamp(0,36)

        tA,tB = dms.match_size(A,B)
        _,tmask = dms.match_size(tB,mask)
        print( pdb_fp, dms.score(tA,tB,tmask) )
