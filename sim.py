import msprime


def one_pulse(g=15, s0=0.5, sample_sizes=[100, 100, 100], mu=1.25e-8, rho = 1.6e-9, N_haploid = [1000, 1000, 1000], lenght_m = 1, seed=1):
    length=int(lenght_m/rho)

    dem = msprime.Demography()
    dem.add_population(name='A', initial_size=N_haploid[0])
    dem.add_population(name='B', initial_size=N_haploid[1])
    dem.add_population(name='C', initial_size=N_haploid[2])
    dem.add_population(name='old', initial_size=1000)

    dem.add_admixture(time=g + 1, ancestral=["A","B"], derived="C", proportions=[1-s0, s0])
    dem.add_population_split(time=5000, derived=["A", "B"], ancestral="old")
    dem.sort_events()

    ts = msprime.sim_ancestry(
        samples={"A": sample_sizes[0], "B": sample_sizes[1], "C": sample_sizes[2]},
        demography=dem, sequence_length = length, recombination_rate=rho, ploidy=2,
        #model=[msprime.DiscreteTimeWrightFisher(duration=50), msprime.StandardCoalescent(duration=3950)],
        #model=[msprime.DiscreteTimeWrightFisher(duration=3950), msprime.StandardCoalescent(duration=50)],
        #model=msprime.DiscreteTimeWrightFisher(),
        #model=msprime.StandardCoalescent(),
        model='hudson',
        random_seed=seed)
    #mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
    return ts
