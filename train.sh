#!/bin/bash

python main.py \
  --json-root /mnt/d/data/medical/image_reports/CTRATE/multi_abnormality_labels \
  --data-root /mnt/d/data/medical/image_reports/CTRATE/train_fixed_process/process_stage2_gz \
  --train-data train_predicted_labels_clean.csv \
  --zeroshot-ct-rate /mnt/d/data/medical/image_reports/CTRATE/multi_abnormality_labels/valid_predicted_labels_clean.csv \
  --report-root /mnt/d/data/medical/image_reports/CTRATE/vqa/train_vqa_MRG_clean.json \
  --name test \
  --warmup 313 \
  --input-data-type stage2 \
  --batch-size 148 \
  --accum-batch 1 \
  --lr=2e-4 \
  --wd=0.2 \
  --epochs 2 \
  --cxr-bert-path /mnt/d/weights/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext 2>&1 | tee output.txt
