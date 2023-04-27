import sys
import numpy as np
from lamom.proportions import kMoment

from simulator.sim import const_gen_flow

seed = int(sys.argv[1])

T_start = 3
duration = 5
total_s = 0.19316375679542194
num_replicates = int(sys.argv[2])

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [1000, 1000, 1000],
                    num_replicates=num_replicates, length_m=2.23)
exp = kMoment(1000, ts=replicates, lengths=[2.23])
k1, k2, k3, k4, lengths = exp.sample_k(unite=False)


for i in range(num_replicates):
    print(f'{k1[i]}\t{k2[i]}\t{k3[i]}\t{k4[i]}')
