import sys

from sim import const_gen_flow
from proportions import kMoment

seed = 12351252

ppg = 0.027
T_start = 10

duration = int(sys.argv[1])
sim_i = int(sys.argv[2])

ts = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration - 1 + 1, sg=ppg, # +1 is for index
                    seed=T_start+duration + 1 + seed + sim_i, N_haploid = [10000, 10000, 10000])
exp = kMoment(10000, ts=ts)
ks = exp.sample_k()
print('random seed\tk1\tk2\tk3')
print(T_start+duration + 1 + seed + sim_i, *ks, sep='\t')
