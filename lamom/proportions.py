import numpy as np
from scipy.stats import kstat
from scipy.optimize import least_squares
from scipy import integrate

from lamom.calcmom import getk1, getk2, getk3, get_prop_per_gen

class kMoment:
    def __init__(self, N=1000, ts=None, lengths=[1]):
        self.N = N

        self.lengths = np.array([lengths], dtype=float)
        if type(lengths) == list:
            lengths.sort()
            self.lengths = np.array(lengths, dtype=float)

        self.ts = ts

    # msprime
    def get_individuals_nodes(self, ts, population, sampled=False):
        nodes = [] # nodes of samples
        for node in ts.nodes(): # they are contained in return of ts.nodes()
            if node.population == population: # population that we need
                if not sampled or node.individual != -1: # TRUE/FALSE + sampled_conditon = TRUE/sampled_conditon
                    nodes.append(node.id)
        return nodes

    # tskit, old version, not working well
    def get_admixture_proportions(self, ts, admixed_population, source_population, length_m=1, rho=1.6e-9):
        length=int(length_m/rho)
        src1_nodes = self.get_individuals_nodes(ts, source_population)
        adm_nodes = self.get_individuals_nodes(ts, admixed_population, sampled=True)
        anc = ts.tables.map_ancestors(adm_nodes, src1_nodes) #table of admixture tracts
        #anc = anc.asdict()
        admixture_proportions = np.zeros(len(adm_nodes))
        for i in range(len(adm_nodes)):
            node_id = adm_nodes[i]
            L = anc[anc.child == node_id].left
            R = anc[anc.child == node_id].right
            proportion = (R - L).sum()
            proportion = proportion / length
            admixture_proportions[i] = proportion
        self.af = admixture_proportions
        return admixture_proportions


    # updated, work well, big variance
    def get_admixture_fractions_bvar(self, replicates, admixed_population, source_population,
                                length_m=1, rho=1.6e-9, time=50):

        props = np.zeros(200)
        total_length = 0
        for replicate_i, ts in enumerate(replicates):
            adm_nodes = self.get_individuals_nodes(ts, admixed_population, sampled=True)
            if replicate_i == 0:
                props = np.zeros(len(adm_nodes))
            LA = np.zeros((len(adm_nodes), ts.num_trees))
            segment_length = np.zeros(ts.num_trees)
            for tree_i, tree in enumerate(ts.trees()):
                segment_length[tree_i] = tree.interval[1] - tree.interval[0]
                for sample_i, node in enumerate(adm_nodes):
                    while tree.time(node) < time:
                        node = tree.parent(node)
                    LA[sample_i, tree_i] = tree.population(node) # 0 - first, 1 - second src
                props[:] += segment_length[tree_i]*LA[:, tree_i]
            total_length += ts.sequence_length
        return props/total_length

    # updated, work well, lower variance
    def get_admixture_moments(self, replicates, admixed_population, source_population,
                                length_m=1, rho=1.6e-9, time=50):

        k1 = 0
        k2 = 0
        k3 = 0
        total_length = 0
        for replicate_i, ts in enumerate(replicates):
            adm_nodes = self.get_individuals_nodes(ts, admixed_population, sampled=True)
            props = np.zeros(len(adm_nodes))
            LA = np.zeros((len(adm_nodes), ts.num_trees))
            segment_length = np.zeros(ts.num_trees)
            for tree_i, tree in enumerate(ts.trees()):
                segment_length[tree_i] = tree.interval[1] - tree.interval[0]
                for sample_i, node in enumerate(adm_nodes):
                    while tree.time(node) < time:
                        node = tree.parent(node)
                    LA[sample_i, tree_i] = tree.population(node) # 0 - first, 1 - second src
                props[:] += segment_length[tree_i]*LA[:, tree_i]
            props /= ts.sequence_length
            k1 += kstat(props, 1)*ts.sequence_length
            k2 += kstat(props, 2)*ts.sequence_length**2
            k3 += kstat(props, 3)*ts.sequence_length**3
            total_length += ts.sequence_length
        k1 /= total_length
        k2 /= total_length**2
        k3 /= total_length**3
        return k1, k2, k3


    def sample_k(self, time=50):
        self.k1_sampled, self.k2_sampled, self.k3_sampled = \
            self.get_admixture_moments(self.ts, 2, 1, time=50)
        print('data moments:')
        print(f'k1: {self.k1_sampled}\nk2: {self.k2_sampled}\nk3: {self.k3_sampled}')
        return self.k1_sampled, self.k2_sampled, self.k3_sampled

    def init_matrixes(self, n, s0): #need to update, legacy
        M4 = np.array([
            [n**2 - 6*n + 6, -2*n, -2*n, -2*n],
            [n - 1, n**2 - n, n, n],
            [n - 1, n, n**2 - n, n],
            [n - 1, n, n, n**2 - n]
        ]) * (n-1) / n**3

        v0 = np.array([1-s0,
                      (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2, (1-s0)**2,
                      (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3, (1-s0)**3,
                      (1-s0)**4])
        y4 = [1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -6]
        y41 = [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, -1, 1]
        y42 = [0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 1]
        y43 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, -1, 0, 0, 1]
        y = np.array([y4, y41, y42, y43], dtype=np.float64)
        return M4, v0, y

    def LD4(self, a, b, c, d, g, *args): #need to update, legacy
        R4 = np.array([
            [np.exp(-2*(np.abs(a-b) + np.abs(b-c) + np.abs(c-d))), 0, 0, 0],
            [0, np.exp(-2*(np.abs(a-b)+np.abs(c-d))), 0, 0],
            [0, 0, np.exp(-2*(np.abs(a-c)+np.abs(b-d))), 0],
            [0, 0, 0, np.exp(-2*(np.abs(a-d)+np.abs(c-b)))]])
        y = args[2]
        v0 = args[1]
        M4 = args[0]
        w0 = y @ v0
        wg = w0
        for i in range(g):
            wg = M4@ (R4 @ w0)
        ld4 = wg[0]
        return ld4

    def eq_k2(self, s, g, L=1, N=10000):
        def integrand(l, g, L):
            return (L-l) * (0.5 + 0.5*np.exp(-2*l))**g
        return 2 * integrate.quad(integrand, 0, L, args=[g, L])[0] * (1-1/2/N)**g * (s - s*s)

    def eq_2(self, g, s, N):
        return (1 - 1 / (2*N))**g * (s - s**2) * 2 / (2 + g)

    def metr_one_pulse(self, g):
        return self.eq_2(g, self.k1_sampled, self.N) - self.k2_sampled

    def metr_cont(self, par):
        prop_per_gen = get_prop_per_gen(self.k1_sampled, par[1])
        if not self.silence:
            print(par)
        return (
            getk2(prop_per_gen, par[0], par[1],
                  self.lengths, N=self.N) - self.k2_sampled,
            getk3(prop_per_gen, par[0], par[1],
                  self.lengths, N=self.N) - self.k3_sampled
            )

    def metr_cont_start_end(self, par):
        prop_per_gen = get_prop_per_gen(self.k1_sampled, par[1]-par[0]+1)
        if not self.silence:
            print(par)
        return (
            getk2(prop_per_gen, par[0], par[1]-par[0]+1,
                  self.lengths, N=self.N) - self.k2_sampled,
            getk3(prop_per_gen, par[0], par[1]-par[0]+1,
                  self.lengths, N=self.N) - self.k3_sampled
            )

    def metr_cont_3(self, par):
        k1 = getk1(par[0], par[2])
        k2 = getk2(prop_per_gen, par[0], par[1], L=self.L, N=self.N)
        k3 = getk3(prop_per_gen, par[0], par[1], L=self.L, N=self.N)
        if not self.silence:
            print(par, k1, k2, k3)
        return (k1 - self.k1_sampled,
                k2 - self.k2_sampled,
                k3 - self.k3_sampled)

    def estimate_one_pulse(self, x0=[5]):
        x = least_squares(self.metr_one_pulse, x0)
        self.res = x
        return x.x

    def estimate_cont(self, silence=True, x0=[5, 5], opb=False):
        """
            return:
                s - int, prop per gen
                T_start - int, generation, when admixture started
                T_end - int, generation, when admixture ended
                cost - int, value of cost function
        """

        self.silence = silence

        if opb: # dur > opt - g_start + 1
            optime = self.estimate_one_pulse()[0]
            print(optime)
            x0 = [optime*0.5, optime*1.5]
            x = least_squares(self.metr_cont_start_end, x0,
                              bounds=([0, optime], [optime, np.inf]))
            T_end = x.x[1]
            duration = x.x[1] - x.x[0] + 1
        else:
            bounds = ([0, 0.05], [np.inf, np.inf])
            x = least_squares(self.metr_cont, x0,
                              bounds=([0, 0.05], [np.inf, np.inf]))
            T_end = x.x[1] + x.x[0] - 1
            duration = x.x[1]
        T_start = x.x[0]
        self.res = x
        return get_prop_per_gen(self.k1_sampled, duration), T_start, T_end, x.cost

    def metr_cont_discr_1(self, T_start, dur, ppg):
        T_start = T_start[0]
        k3 = getk3(ppg, T_start, dur, L=self.L, N=self.N)
        return (k3 - self.k3_sampled)

    def metr_cont_discr_2(self, dur, T_start):
        ppg = get_prop_per_gen(self.k1_sampled, dur)
        k2 = getk2(ppg, T_start, dur[0], L=self.L, N=self.N)
        return (k2 - self.k2_sampled)

    def estimate_cont_discr(self, n_it=20, T_start0=10, dur0=5):
        estimates = []
        costs = []
        T_start_next_next = T_start0
        dur_next = dur0
        it = 0
        while it < n_it:
            ores = least_squares(self.metr_cont_discr_2, [dur_next],
                              bounds=(0, np.inf), args=[T_start_next_next])
            dur_next = ores.x[0]
            dur_next_cost = ores.cost
            print([T_start_next_next, dur_next])
            ppg = get_prop_per_gen(self.k1_sampled, dur_next)
            ores = least_squares(self.metr_cont_discr_1, [T_start_next_next],
                              bounds=(0, np.inf), args=[dur_next, ppg])
            T_start_next_next = ores.x[0]
            T_start_next_next_cost = ores.cost
            estimates.append([dur_next, T_start_next_next])
            costs.append([dur_next_cost, T_start_next_next_cost])
            it+=1
        return estimates, costs

    def estimate_cont_3(self, silence=True):

        self.silence = silence

        x0 = [get_prop_per_gen(self.k1_sampled, 5), 5, 5]
        x = least_squares(self.metr_cont, x0,
                          bounds=([0, 0, 0], [self.k1_sampled, np.inf, np.inf]))
        self.res = x
        return x.x[0], x.x[1], x.x[2]
