import sys
import numpy as np
from lamom.proportions import kMoment

from simulator.sim import const_gen_flow

seed = int(sys.argv[1])

T_start = 5
duration = 5
total_s = 0.3
num_replicates = 2

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [1000, 1000, 1000],
                    num_replicates=num_replicates)
exp = kMoment(1000, ts=replicates, lengths=[1]*num_replicates)
k1_u, k2_u, k3_u, lengths_u = exp.sample_k(unite=True)
exp.make_batch(k1_u, k2_u, k3_u, lengths_u, batchsize=1)

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [1000, 1000, 1000],
                    num_replicates=num_replicates)
exp = kMoment(1000, ts=replicates, lengths=[1]*num_replicates)
k1_nu, k2_nu, k3_nu, lengths_nu = exp.sample_k(unite=False)
exp.make_batch(k1_nu, k2_nu, k3_nu, lengths_nu, batchsize=num_replicates)
