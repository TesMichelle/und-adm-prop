import sys
import numpy as np
from lamom.proportions import kMoment

from simulator.sim import const_gen_flow

seed = int(sys.argv[1])
length_m = float(sys.argv[2])

T_start = int(sys.argv[3])
duration = int(sys.argv[4])
total_s = 0.19316375679542194
num_replicates = 2

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [1000, 1000, 1000],
                    num_replicates=num_replicates, length_m=length_m,
                    sample_sizes=[51, 52, 21])
exp = kMoment(1000, ts=replicates, lengths=[length_m, length_m])
k1, k2, k3, k4, lengths = exp.sample_k(unite=True)


print(f'{k1[0]}\t{k2[0]}\t{k3[0]}\t{k4[0]}')
