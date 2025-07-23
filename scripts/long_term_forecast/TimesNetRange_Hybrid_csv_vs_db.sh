#!/bin/bash
# Example comparison script for TimesNetRange_Hybrid using CSVFolder and SQLiteFolder loaders

export CUDA_VISIBLE_DEVICES=0
model_name=TimesNetRange_Hybrid

# ---------- CSVFolder ----------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --data csvfolder \
  --root_path /path/to/csv/folder/ \
  --state all \
  --model_id csv_test \
  --model $model_name \
  --features MS \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 30 \
  --enc_in 10 --dec_in 10 --c_out 10 \
  --interval_mult 2.0 \
  --itr 1

# ---------- SQLiteFolder ----------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --data sqlitefolder \
  --root_path /path/to/db/folder/ \
  --table_name your_table \
  --state all \
  --model_id db_test \
  --model $model_name \
  --features MS \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 30 \
  --enc_in 10 --dec_in 10 --c_out 10 \
  --interval_mult 2.0 \
  --itr 1

