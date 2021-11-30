# generates a distance map and confidence map for custom model given a sequence
# 

import argparse
import os
import dist_map_score as dms
import util
import torch
from os import path


import torch
import esm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, help="sequence", required=True)
    parser.add_argument('--out', type=str, help="output folder", required=True)
    args = parser.parse_args()

    
    if not os.path.isfile( args.out ):
      os.mkdir( args.out )


    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    from ax_model_conv import ax_model
    model = ax_model()
    device = torch.device('cpu')
    st_dict = torch.load('./esm_test_model4.pth', map_location=device)
    model.load_state_dict(st_dict)


    seq = args.seq
    x = [(None,seq)]
    batch_converter = alphabet.get_batch_converter()
    _,_,x =  batch_converter(x)
    x
    esm = esm_model(x, repr_layers=[33], need_head_weights=True)
    reps = esm["representations"][33]
    esm = esm['attentions'][:,:,:,1:-1, 1:-1]


    SF = 4 # Scale Factor
    L = SF- esm.shape[-1]%SF
    esm = torch.nn.functional.pad(esm,(0,L,0,L)) 
    #msks = torch.nn.functional.pad(msks, (0,L))

    x = x[:,1:-1]

    py = model(esm)
    py.shape
    pdistm = py.argmax(dim=1)[0]

    torch.save( (py>.5).sum(dim=1)[0].bool(), args.out + '/confidence_mask' )
    torch.save( pdistm, args.out + '/dist' )
