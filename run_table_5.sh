#!/bin/bash
# table 5: comparison on test set of the paper

datasets=("airfoil" "Bikeshare_DC" "house_kc" "Housing_Boston" "Openml_586" "Openml_589" "Openml_607" "Openml_616" "Openml_618" "Openml_620" "Openml_637" "adult" "amazon_employee" "default_credit_card" "credit_a" "fertility_Diagnosis" "german_credit_24" "hepatitis" "ionosphere" "lymphography" "megawatt1" "messidor_features" "PimaIndian" "spambase" "SPECTF" "winequality-red" "winequality-white")

for dataset in "${datasets[@]}"
do
  python main_attention.py --file_name $dataset --cuda 3 --seed 0 --steps_num 3 --epochs 300 --episodes 24 --split_train_test True --shuffle True --train_size 0.7 &
  python main_attention.py --file_name $dataset --cuda 4 --seed 1 --steps_num 3 --epochs 300 --episodes 24 --split_train_test True --shuffle True --train_size 0.7 &
  python main_attention.py --file_name $dataset --cuda 5 --seed 2 --steps_num 3 --epochs 300 --episodes 24 --split_train_test True --shuffle True --train_size 0.7 &
  python main_attention.py --file_name $dataset --cuda 6 --seed 3 --steps_num 3 --epochs 300 --episodes 24 --split_train_test True --shuffle True --train_size 0.7 &
  python main_attention.py --file_name $dataset --cuda 7 --seed 4 --steps_num 3 --epochs 300 --episodes 24 --split_train_test True --shuffle True --train_size 0.7 &
  wait
done
