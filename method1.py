import argparse
import dist_map_score as dms
import util
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, help="path to pdb", required=True)
    args = parser.parse_args()




    distm_fp = 'af2_distm/dist'
    mask_fp = 'af2_distm/confidence_mask'
    pdb_fp = args.pdb

    A = torch.load(distm_fp)
    mask = torch.load(mask_fp)
    B = util.distm_pdb(pdb_fp)
    A,B = dms.match_size(A,B)
    _,mask = dms.match_size(A,mask)
    print( dms.score(A,B,mask) )
