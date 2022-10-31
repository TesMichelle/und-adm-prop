import sys

from proportions import kMoment
from sim import const_gen_flow

ind = int(sys.argv[1])

i = ind % 4
j = (ind // 4) % 4
k = (ind // 4) // 4

T_start_vals = [2, 5, 10, 15]
dur_vals = [2, 5, 10, 15]
ppg_vals = [0.02, 0.03, 0.04, 0.05]

T_start = T_start_vals[i]
dur = dur_vals[j]
ppg = ppg_vals[k]

sim_num = 10

print('True:', ppg, T_start, dur)

for sim_i in range(sim_num):
    ts = const_gen_flow(g_start=T_start - 1, g_end=T_start+dur-1 - 1, sg=ppg, # -1 is for index
                        seed=ind*sim_num + sim_i + 1)
    exp = kMoment(1000, ts=ts)
    exp.sample_k()
    print(*exp.estimate_cont(silence=True))
