import sys
import msprime
from scipy.stats import kstat

from proportions import kMoment
from sim import const_gen_flow

T = int(sys.argv[1])
size = int(sys.argv[2])
sg = float(sys.argv[3])


T_start = T // size
dur = T % size

sim_num = 100

k1 = 0
k2 = 0
k3 = 0
for sim_i in range(sim_num):
    ts = const_gen_flow(g_start=T_start - 1, g_end=T_start+dur-1 - 1, sg=sg, # -1 is for index
                        seed=sim_num*size*T_start+sim_num*dur+sim_i + 1)
    exp = kMoment(1000, ts=ts)
    ap = exp.get_admixture_proportions(ts, 2, 1)

    k1 += kstat(ap, 1)
    k2 += kstat(ap, 2)
    k3 += kstat(ap, 3)
print(k1/sim_num, k2/sim_num, k3/sim_num)
