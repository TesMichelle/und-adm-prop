import sys
import numpy as np
from lamom.proportions import kMoment

from simulator.sim import const_gen_flow

seed = int(sys.argv[1])
length_m = float(sys.argv[2])

T_start = int(sys.argv[3])
duration = int(sys.argv[4])
total_s = 0.5
num_replicates = 2

replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [1000, 1000, 1000],
                    num_replicates=num_replicates, length_m=length_m,
                    sample_sizes=[51, 52, 21])
exp = kMoment(1000, ts=replicates, lengths=[length_m, length_m])

for replicate_i, ts in enumerate(replicates):
    if replicate_i == 0:
        adm_nodes = exp.get_individuals_nodes(ts, 2, sampled=True)
        props_overall = np.zeros(len(adm_nodes))
    props_overall += exp.get_admixture_proportions(ts, 2,
        1, 50) * ts.sequence_length
    total_length += ts.sequence_length
props_overall /= total_length

print(props_overall)
