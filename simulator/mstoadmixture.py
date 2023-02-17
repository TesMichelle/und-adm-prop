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
                    num_replicates=num_replicates, sim_mut=True)

ind_names = [f'Adm_{i}' for i in range(1, 101)] +\
            [f'Src1_{i}' for i in range(1, 101)] +\
            [f'Src2_{i}' for i in range(1, 101)]

dir = f'adm_test/seed_{seed}'
os.mkdir(dir)
with open(dir+'/data.pop', 'w') as f:
    for i in range(300):
        if i < 100:
            f.write('-\n')
        elif i < 200:
            f.write('SRC1\n')
        else:
            f.write('SRC2\n')

for i, ts in tqdm(enumerate(replicates)):
    vcf_file_path = f'adm_test/seed_{seed}/chr_{i+1}.vcf'
    with open(vcf_file_path, "wt") as vcf_file:
        ts.write_vcf(vcf_file, individual_names = ind_names, contig_id=str(i+1))
        vcf_file.close()
        subprocess.call(['./adm_test/zipper.sh', vcf_file_path])
        subprocess.call(["rm", vcf_file_path])

print('merging chromosomes in one .vcf ...')
subprocess.call(["./adm_test/merger.sh", str(seed)])
