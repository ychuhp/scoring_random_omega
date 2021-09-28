import argparse
import math


def score(A,B,mask):
    mask = mask.float()
    A,B = A.float(), B.float()
    D = ((A-B)*mask)**2
    avg = D.sum()/mask.sum()
    return math.sqrt(avg)


def match_size(A,B):
    x = A.shape[0]
    y = B.shape[0]
    k = abs(x-y)

    if x>y:
        A = A[:-k, :-k]
    elif x<y:
        B = B[:-k, :-k]

    return A,B


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('distm', type=str, help="path to distance matrix", required=True)
    parser.add_argument('mask', type=str, help="path to distance matrix confidence mask", required=True)
    parser.add_argument('pdb', type=str, help="path to pdb", required=True)
    args = parser.parse_args()
