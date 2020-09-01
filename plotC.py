#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse, sys
from scipy.stats import pearsonr

from mi3gpu.utils.potts_common import indepF

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bimA')
    parser.add_argument('bimB')
    parser.add_argument('--name', default='C')
    args = parser.parse_args(sys.argv[1:])

    ba = np.load(args.bimA)
    ca = ba - indepF(ba)

    bb = np.load(args.bimB)
    cb = bb - indepF(bb)

    r = pearsonr(ca.ravel(), cb.ravel())[0]
    print("rho:", r)
    print("sum abs-err:", np.sum(np.abs(ca-cb)))

    plt.plot(ca.ravel(), cb.ravel(), ',')
    mi, ma = np.min(ca), np.max(ca)
    plt.plot([mi, ma], [mi, ma], 'k-')
    plt.title(r'{}  $\rho = {:.3f}$'.format(args.name, r))
    plt.savefig(args.name)
    
if __name__ == '__main__':
    main()

