#!/usr/bin/env bash


python simple_pair_rag_pipeline/07_trainval_joint_rag_predict_test_fast.py \
  --input_json data/simple_pair_rag_annotations/02b_simple_rag_canonical_dualhead.json \
  --out_json data/simple_pair_rag_annotations/07_trainval_joint_rag_test_r2gen_ready.json \
  --summary_json results/simple_pair_rag_trainval_joint/summary.json \
  2>&1 | tee logs/simple_pair_rag/trainval_joint_rag.log
