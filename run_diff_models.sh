#!/bin/bash
# comparison on different models of the paper

models=("lgb" "xgb" "lr" "cat" "rf")

for m in "${models[@]}"
do
  python main_attention.py --file_name german_credit_24 --cuda 4 --seed 1 --steps_num 3 --epochs 300 --episodes 24 --model $m &
  python main_attention.py --file_name hepatitis --cuda 1 --seed 1 --steps_num 3 --epochs 300 --episodes 24 --model $m &
  python main_attention.py --file_name ionosphere --cuda 1 --seed 1 --steps_num 3 --epochs 300 --episodes 24 --model $m &
  python main_attention.py --file_name messidor_features --cuda 4 --seed 1 --steps_num 3 --epochs 300 --episodes 24 --model $m &
  wait
done
