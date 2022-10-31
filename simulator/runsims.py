import sis

from sim import const_gen_flow
from proportions import one_pulse_estimator

g_start = int(sys.argv[0])
max_g_end = 5
filename = "props_g_start_" + g_start

sim_num = 100
sample_sizes = [100, 100, 100]
sg = 0.2

props = np.zeros((max_g_end-g_start, sim_num, 2*sample_sizes[0]))
for g_end in tqdm(range(g_start+1, max_g_end+1)):
    for sim_i in range(sim_num):
        ts = const_gen_flow(
            g_start=g_start, g_end=g_end, sg=sg,
            sample_sizes = sample_sizes,
            seed=10000*g_start+100*g_end+sim_i
        )
        exp = one_pulse_estimator(1000, ts=ts)
        props[g_end-g_start-1][sim_i] = exp.get_admixture_proportions(ts, 2, 0)
np.save(filename, props)
