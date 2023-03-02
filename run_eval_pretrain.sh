#!/bin/bash
# evaluate pre-trained models of the paper

datasets=("winequality-red" "Housing_Boston" "Openml_616" "Openml_637" "ionosphere")

for dataset in "${datasets[@]}"
do
  python main_attention.py --file_name $dataset --cuda 4 --seed 0 --steps_num 2 --epochs 300 --episodes 24 &
  python main_attention.py --file_name $dataset --cuda 5 --seed 0 --steps_num 2 --epochs 300 --episodes 24 --enc_c_pth ./pretrained_models/enc_c_param_oml.pkl --enc_d_pth ./pretrained_models/enc_d_param_oml.pkl &
  python main_attention.py --file_name $dataset --cuda 6 --seed 0 --steps_num 2 --epochs 300 --episodes 24 --enc_c_pth ./pretrained_models/enc_c_param_uci.pkl --enc_d_pth ./pretrained_models/enc_d_param_uci.pkl &
  python main_attention.py --file_name $dataset --cuda 7 --seed 0 --steps_num 2 --epochs 300 --episodes 24 --enc_c_pth ./pretrained_models/enc_c_param_mix.pkl --enc_d_pth ./pretrained_models/enc_d_param_mix.pkl &
  wait
done
