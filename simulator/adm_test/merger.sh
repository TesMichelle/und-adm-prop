#!/bin/sh

seed=$1

bcftools concat adm_test/seed_${seed}/seed_${seed}_chr_{1..2}.vcf.gz | bgzip -c > adm_test/seed_${seed}/seed_${seed}_all.vcf.gz
tabix -C -p vcf adm_test/seed_${seed}/seed_${seed}_all.vcf.gz
