import numpy as np
from scipy.stats import kstat
from scipy.optimize import least_squares

class one_pulse_estimator:
    def __init__(self, N, af=None, ts=None):
        if af == None:
            af = self.get_admixture_proportions(ts, 2, 0)
        self.k2 = kstat(af)
        self.s0 = np.mean(af)
        self.N = N
        self.ts = ts

    def get_individuals_nodes(self, ts, population, sampled=False):
        nodes = [] # nodes of samples
        for node in ts.nodes(): # they are contained in return of ts.nodes()
            if node.population == population: # population that we need
                if not sampled or node.individual != -1: # TRUE/FALSE + sampled_conditon = TRUE/sampled_conditon
                    nodes.append(node.id)
        return nodes

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
            T = np.c_[L, R]
            proportion = (T[:,1] - T[:, 0]).sum()
            proportion = proportion / length
            admixture_proportions[i] = proportion
        self.af = admixture_proportions
        return admixture_proportions

    def init_matrixes(self, n, s0): #4 moment
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

    def LD4(self, a, b, c, d, g, *args): #4 moment
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

    def eq_2(self, g, s, N):
        return (1 - 1 / (2*N))**g * (s - s**2) * 2 / (2 + g)

    def metr(self, g):
        return self.eq_2(g, self.s0, self.N) - self.k2

    def estimate(self):
        x0 = 1
        x = least_squares(self.metr, x0)
        self.res = x
        return x.x
