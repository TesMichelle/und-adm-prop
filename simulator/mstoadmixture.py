from sim import const_gen_flow

import os
import sys
from tqdm import tqdm
import subprocess

N = 1000
seed = int(sys.argv[1])
T_start = 5
duration = 5
total_s = 0.3
num_replicates = 2


replicates = const_gen_flow(g_start=T_start + 1, g_end=T_start+duration, total_s=total_s, # +1 is for index
                    seed=T_start+duration + 1 + seed, N_haploid = [N, N, N],
                    num_replicates=num_replicates, sim_mut=True, mu=1e-9)

ind_names = [f'SrcA{i}' for i in range(1, 101)] +\
            [f'SrcB{i}' for i in range(1, 101)] +\
            [f'AdmC{i}' for i in range(1, 101)]

dir = f'adm_test/seed_{seed}'
os.mkdir(dir)

for i, ts in tqdm(enumerate(replicates)):
    vcf_file_path = f'adm_test/seed_{seed}/seed_{seed}_chr_{i+1}.vcf'
    with open(dir+f'/seed_{seed}_chr_{i+1}.vcf.gz.pop', 'w') as f:
        for j in range(300):
            if j < 100:
                f.write('SRCA\n')
            elif j < 200:
                f.write('SRCB\n')
            else:
                f.write('-\n')
    with open(vcf_file_path, "wt") as vcf_file:
        ts.write_vcf(vcf_file, individual_names = ind_names, contig_id=str(i+1))
        vcf_file.close()
        subprocess.call(['./adm_test/zipper.sh', vcf_file_path])
        subprocess.call(["rm", vcf_file_path])

print('merging chromosomes in one .vcf ...')
subprocess.call(["./adm_test/merger.sh", str(seed)])
with open(dir+f'/seed_{seed}_all.vcf.gz.pop', 'w') as f:
    for j in range(300):
        if j < 100:
            f.write('SRCA\n')
        elif j < 200:
            f.write('SRCB\n')
        else:
            f.write('-\n')
