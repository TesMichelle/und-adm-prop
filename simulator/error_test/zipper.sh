#!/bin/sh
vcffile=$1
bgzip  -c ${vcffile} > ${vcffile}.gz
tabix -C -p vcf ${vcffile}.gz
