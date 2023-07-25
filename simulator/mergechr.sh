#!/bin/sh

ts=3
dur=6

grep -n '.' *.log | grep -e :1: | cut -d: -f3- > chr1_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :2: | cut -d: -f3- > chr2_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :3: | cut -d: -f3- > chr3_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :4: | cut -d: -f3- > chr4_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :5: | cut -d: -f3- > chr5_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :6: | cut -d: -f3- > chr6_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :7: | cut -d: -f3- > chr7_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :8: | cut -d: -f3- > chr8_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :9: | cut -d: -f3- > chr9_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :10: | cut -d: -f3- > chr10_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :11: | cut -d: -f3- > chr11_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :12: | cut -d: -f3- > chr12_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :13: | cut -d: -f3- > chr13_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :14: | cut -d: -f3- > chr14_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :15: | cut -d: -f3- > chr15_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :16: | cut -d: -f3- > chr16_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :17: | cut -d: -f3- > chr17_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :18: | cut -d: -f3- > chr18_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :19: | cut -d: -f3- > chr19_${ts}_${dur}.txt
grep -n '.' *.log | grep -e :20: | cut -d: -f3- > chr20_${ts}_${dur}.txt
