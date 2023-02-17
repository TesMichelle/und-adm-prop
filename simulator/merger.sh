#!/bin/sh

bcftools concat chr_{1..20}.vcf | bgzip -c > data.vcf.gz
tabix -C -p vcf data.vcf.gz
