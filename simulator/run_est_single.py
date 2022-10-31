import sys

from sim import const_gen_flow
from proportions import kMoment

seed = 12351252

ppg = 0.027
T_start = 2
duration = 11

sim_i = int(sys.argv[1])

ts = const_gen_flow(g_start=T_start - 1, g_end=T_start+duration -1 - 1, sg=ppg, # -1 is for index
                    seed=T_start+duration + 1 + seed + sim_i, N_haploid = [10000, 10000, 10000])
exp = kMoment(10000, ts=ts)
exp.sample_k()
_ = exp.estimate_cont(silence=True)
print(*_)
