#!/usr/bin/env bash
set -ux
cd /mnt/e/development/mmcif-vector-db
FS=/root/foldseek/foldseek/bin/foldseek
mkdir -p results_v3/fs_tmp
rm -rf results_v3/fs_db results_v3/fs_out.tsv
$FS createdb v3/domains results_v3/fs_db --threads 8 2>&1 | tail -n 20
$FS easy-search results_v3/fs_db results_v3/fs_db results_v3/fs_out_raw.tsv results_v3/fs_tmp \
  --exhaustive-search 1 -e 1000 --max-seqs 10000 --threads 8 \
  --format-output query,target,alntmscore,qtmscore,ttmscore,bits,evalue 2>&1 | tail -n 40
echo "raw rows:"; wc -l results_v3/fs_out_raw.tsv; head -n 3 results_v3/fs_out_raw.tsv
printf "query\ttarget\talntmscore\tqtmscore\tttmscore\tbits\tevalue\n" > results_v3/foldseek_gold.tsv
cat results_v3/fs_out_raw.tsv >> results_v3/foldseek_gold.tsv
echo "gold rows (incl header):"; wc -l results_v3/foldseek_gold.tsv
