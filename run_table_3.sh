#!/bin/bash
# table 3: comparison on large-scale datasets of the paper

datasets=("medical_charges" "poker_hand" "GiveMeCredit")

for dataset in "${datasets[@]}"
do
  python main_attention.py --file_name $dataset --cuda 3 --seed 0 --steps_num 3 --epochs 300 --episodes 24 &
  python main_attention.py --file_name $dataset --cuda 4 --seed 1 --steps_num 3 --epochs 300 --episodes 24 &
  python main_attention.py --file_name $dataset --cuda 5 --seed 2 --steps_num 3 --epochs 300 --episodes 24 &
  python main_attention.py --file_name $dataset --cuda 6 --seed 3 --steps_num 3 --epochs 300 --episodes 24 &
  python main_attention.py --file_name $dataset --cuda 7 --seed 4 --steps_num 3 --epochs 300 --episodes 24 &
  wait
done

