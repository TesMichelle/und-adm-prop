import sys

from simulator.sim import const_gen_flow
from lamom.proportions import kMoment



seed = int(sys.argv[1])

T_start = 2
duration = 10
total_s = 0.3

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [10000, 10000, 10000],
                    num_replicates=2)

exp = kMoment(10000, ts=replicates)
exp.sample_k()
exp.estimate_cont(x0=[5, 5], silence=True)
